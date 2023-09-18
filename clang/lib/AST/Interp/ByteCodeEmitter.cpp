//===--- ByteCodeEmitter.cpp - Instruction emitter for the VM ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ByteCodeEmitter.h"
#include "Context.h"
#include "Floating.h"
#include "Opcode.h"
#include "Program.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/DeclCXX.h"
#include <type_traits>

using namespace clang;
using namespace clang::interp;

using APSInt = llvm::APSInt;
using Error = llvm::Error;

Expected<Function *>
ByteCodeEmitter::compileFunc(const FunctionDecl *FuncDecl) {
  // Set up argument indices.
  unsigned ParamOffset = 0;
  SmallVector<PrimType, 8> ParamTypes;
  SmallVector<unsigned, 8> ParamOffsets;
  llvm::DenseMap<unsigned, Function::ParamDescriptor> ParamDescriptors;

  // If the return is not a primitive, a pointer to the storage where the
  // value is initialized in is passed as the first argument. See 'RVO'
  // elsewhere in the code.
  QualType Ty = FuncDecl->getReturnType();
  bool HasRVO = false;
  if (!Ty->isVoidType() && !Ctx.classify(Ty)) {
    HasRVO = true;
    ParamTypes.push_back(PT_Ptr);
    ParamOffsets.push_back(ParamOffset);
    ParamOffset += align(primSize(PT_Ptr));
  }

  // If the function decl is a member decl, the next parameter is
  // the 'this' pointer. This parameter is pop()ed from the
  // InterpStack when calling the function.
  bool HasThisPointer = false;
  if (const auto *MD = dyn_cast<CXXMethodDecl>(FuncDecl)) {
    if (MD->isInstance()) {
      HasThisPointer = true;
      ParamTypes.push_back(PT_Ptr);
      ParamOffsets.push_back(ParamOffset);
      ParamOffset += align(primSize(PT_Ptr));
    }

    // Set up lambda capture to closure record field mapping.
    if (isLambdaCallOperator(MD)) {
      const Record *R = P.getOrCreateRecord(MD->getParent());
      llvm::DenseMap<const ValueDecl *, FieldDecl *> LC;
      FieldDecl *LTC;

      MD->getParent()->getCaptureFields(LC, LTC);

      for (auto Cap : LC) {
        unsigned Offset = R->getField(Cap.second)->Offset;
        this->LambdaCaptures[Cap.first] = {
            Offset, Cap.second->getType()->isReferenceType()};
      }
      // FIXME: LambdaThisCapture
      (void)LTC;
    }
  }

  // Assign descriptors to all parameters.
  // Composite objects are lowered to pointers.
  for (const ParmVarDecl *PD : FuncDecl->parameters()) {
    std::optional<PrimType> T = Ctx.classify(PD->getType());
    PrimType PT = T.value_or(PT_Ptr);
    Descriptor *Desc = P.createDescriptor(PD, PT);
    ParamDescriptors.insert({ParamOffset, {PT, Desc}});
    Params.insert({PD, {ParamOffset, T != std::nullopt}});
    ParamOffsets.push_back(ParamOffset);
    ParamOffset += align(primSize(PT));
    ParamTypes.push_back(PT);
  }

  // Create a handle over the emitted code.
  Function *Func = P.getFunction(FuncDecl);
  if (!Func)
    Func = P.createFunction(FuncDecl, ParamOffset, std::move(ParamTypes),
                            std::move(ParamDescriptors),
                            std::move(ParamOffsets), HasThisPointer, HasRVO);

  assert(Func);
  // For not-yet-defined functions, we only create a Function instance and
  // compile their body later.
  if (!FuncDecl->isDefined())
    return Func;

  // Lambda static invokers are a special case that we emit custom code for.
  bool IsEligibleForCompilation = false;
  if (const auto *MD = dyn_cast<CXXMethodDecl>(FuncDecl))
    IsEligibleForCompilation = MD->isLambdaStaticInvoker();
  if (!IsEligibleForCompilation)
    IsEligibleForCompilation = FuncDecl->isConstexpr();

  // Compile the function body.
  if (!IsEligibleForCompilation || !visitFunc(FuncDecl)) {
    // Return a dummy function if compilation failed.
    if (BailLocation)
      return llvm::make_error<ByteCodeGenError>(*BailLocation);
    else {
      Func->setIsFullyCompiled(true);
      return Func;
    }
  } else {
    // Create scopes from descriptors.
    llvm::SmallVector<Scope, 2> Scopes;
    for (auto &DS : Descriptors) {
      Scopes.emplace_back(std::move(DS));
    }

    // Set the function's code.
    Func->setCode(NextLocalOffset, std::move(Code), std::move(SrcMap),
                  std::move(Scopes), FuncDecl->hasBody());
    Func->setIsFullyCompiled(true);
    return Func;
  }
}

Scope::Local ByteCodeEmitter::createLocal(Descriptor *D) {
  NextLocalOffset += sizeof(Block);
  unsigned Location = NextLocalOffset;
  NextLocalOffset += align(D->getAllocSize());
  return {Location, D};
}

void ByteCodeEmitter::emitLabel(LabelTy Label) {
  const size_t Target = Code.size();
  LabelOffsets.insert({Label, Target});

  if (auto It = LabelRelocs.find(Label);
      It != LabelRelocs.end()) {
    for (unsigned Reloc : It->second) {
      using namespace llvm::support;

      // Rewrite the operand of all jumps to this label.
      void *Location = Code.data() + Reloc - align(sizeof(int32_t));
      assert(aligned(Location));
      const int32_t Offset = Target - static_cast<int64_t>(Reloc);
      endian::write<int32_t, endianness::native, 1>(Location, Offset);
    }
    LabelRelocs.erase(It);
  }
}

int32_t ByteCodeEmitter::getOffset(LabelTy Label) {
  // Compute the PC offset which the jump is relative to.
  const int64_t Position =
      Code.size() + align(sizeof(Opcode)) + align(sizeof(int32_t));
  assert(aligned(Position));

  // If target is known, compute jump offset.
  if (auto It = LabelOffsets.find(Label);
      It != LabelOffsets.end())
    return It->second - Position;

  // Otherwise, record relocation and return dummy offset.
  LabelRelocs[Label].push_back(Position);
  return 0ull;
}

bool ByteCodeEmitter::bail(const SourceLocation &Loc) {
  if (!BailLocation)
    BailLocation = Loc;
  return false;
}

/// Helper to write bytecode and bail out if 32-bit offsets become invalid.
/// Pointers will be automatically marshalled as 32-bit IDs.
template <typename T>
static void emit(Program &P, std::vector<std::byte> &Code, const T &Val,
                 bool &Success) {
  size_t Size;

  if constexpr (std::is_pointer_v<T>)
    Size = sizeof(uint32_t);
  else
    Size = sizeof(T);

  if (Code.size() + Size > std::numeric_limits<unsigned>::max()) {
    Success = false;
    return;
  }

  // Access must be aligned!
  size_t ValPos = align(Code.size());
  Size = align(Size);
  assert(aligned(ValPos + Size));
  Code.resize(ValPos + Size);

  if constexpr (!std::is_pointer_v<T>) {
    new (Code.data() + ValPos) T(Val);
  } else {
    uint32_t ID = P.getOrCreateNativePointer(Val);
    new (Code.data() + ValPos) uint32_t(ID);
  }
}

template <>
void emit(Program &P, std::vector<std::byte> &Code, const Floating &Val,
          bool &Success) {
  size_t Size = Val.bytesToSerialize();

  if (Code.size() + Size > std::numeric_limits<unsigned>::max()) {
    Success = false;
    return;
  }

  // Access must be aligned!
  size_t ValPos = align(Code.size());
  Size = align(Size);
  assert(aligned(ValPos + Size));
  Code.resize(ValPos + Size);

  Val.serialize(Code.data() + ValPos);
}

template <typename... Tys>
bool ByteCodeEmitter::emitOp(Opcode Op, const Tys &... Args, const SourceInfo &SI) {
  bool Success = true;

  // The opcode is followed by arguments. The source info is
  // attached to the address after the opcode.
  emit(P, Code, Op, Success);
  if (SI)
    SrcMap.emplace_back(Code.size(), SI);

  // The initializer list forces the expression to be evaluated
  // for each argument in the variadic template, in order.
  (void)std::initializer_list<int>{(emit(P, Code, Args, Success), 0)...};

  return Success;
}

bool ByteCodeEmitter::jumpTrue(const LabelTy &Label) {
  return emitJt(getOffset(Label), SourceInfo{});
}

bool ByteCodeEmitter::jumpFalse(const LabelTy &Label) {
  return emitJf(getOffset(Label), SourceInfo{});
}

bool ByteCodeEmitter::jump(const LabelTy &Label) {
  return emitJmp(getOffset(Label), SourceInfo{});
}

bool ByteCodeEmitter::fallthrough(const LabelTy &Label) {
  emitLabel(Label);
  return true;
}

//===----------------------------------------------------------------------===//
// Opcode emitters
//===----------------------------------------------------------------------===//

#define GET_LINK_IMPL
#include "Opcodes.inc"
#undef GET_LINK_IMPL
