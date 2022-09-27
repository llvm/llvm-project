//===--- ByteCodeEmitter.cpp - Instruction emitter for the VM ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ByteCodeEmitter.h"
#include "Context.h"
#include "Opcode.h"
#include "Program.h"
#include "clang/AST/DeclCXX.h"
#include <type_traits>

using namespace clang;
using namespace clang::interp;

using APSInt = llvm::APSInt;
using Error = llvm::Error;

Expected<Function *> ByteCodeEmitter::compileFunc(const FunctionDecl *F) {
  // Do not try to compile undefined functions.
  if (!F->isDefined(F) || (!F->hasBody() && F->willHaveBody()))
    return nullptr;

  // Set up argument indices.
  unsigned ParamOffset = 0;
  SmallVector<PrimType, 8> ParamTypes;
  llvm::DenseMap<unsigned, Function::ParamDescriptor> ParamDescriptors;

  // If the return is not a primitive, a pointer to the storage where the value
  // is initialized in is passed as the first argument.
  QualType Ty = F->getReturnType();
  if (!Ty->isVoidType() && !Ctx.classify(Ty)) {
    ParamTypes.push_back(PT_Ptr);
    ParamOffset += align(primSize(PT_Ptr));
  }

  // Assign descriptors to all parameters.
  // Composite objects are lowered to pointers.
  for (const ParmVarDecl *PD : F->parameters()) {
    PrimType Ty;
    if (llvm::Optional<PrimType> T = Ctx.classify(PD->getType())) {
      Ty = *T;
    } else {
      Ty = PT_Ptr;
    }

    Descriptor *Desc = P.createDescriptor(PD, Ty);
    ParamDescriptors.insert({ParamOffset, {Ty, Desc}});
    Params.insert({PD, ParamOffset});
    ParamOffset += align(primSize(Ty));
    ParamTypes.push_back(Ty);
  }

  // Create a handle over the emitted code.
  Function *Func = P.createFunction(F, ParamOffset, std::move(ParamTypes),
                                    std::move(ParamDescriptors));
  // Compile the function body.
  if (!F->isConstexpr() || !visitFunc(F)) {
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
                  std::move(Scopes));
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
  auto It = LabelRelocs.find(Label);
  if (It != LabelRelocs.end()) {
    for (unsigned Reloc : It->second) {
      using namespace llvm::support;

      /// Rewrite the operand of all jumps to this label.
      void *Location = Code.data() + Reloc - sizeof(int32_t);
      const int32_t Offset = Target - static_cast<int64_t>(Reloc);
      endian::write<int32_t, endianness::native, 1>(Location, Offset);
    }
    LabelRelocs.erase(It);
  }
}

int32_t ByteCodeEmitter::getOffset(LabelTy Label) {
  // Compute the PC offset which the jump is relative to.
  const int64_t Position = Code.size() + sizeof(Opcode) + sizeof(int32_t);

  // If target is known, compute jump offset.
  auto It = LabelOffsets.find(Label);
  if (It != LabelOffsets.end()) {
    return It->second - Position;
  }

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
static void emit(Program &P, std::vector<char> &Code, const T &Val,
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

  if constexpr (!std::is_pointer_v<T>) {
    const char *Data = reinterpret_cast<const char *>(&Val);
    Code.insert(Code.end(), Data, Data + Size);
  } else {
    uint32_t ID = P.getOrCreateNativePointer(Val);
    const char *Data = reinterpret_cast<const char *>(&ID);
    Code.insert(Code.end(), Data, Data + Size);
  }
}

template <typename... Tys>
bool ByteCodeEmitter::emitOp(Opcode Op, const Tys &... Args, const SourceInfo &SI) {
  bool Success = true;

  /// The opcode is followed by arguments. The source info is
  /// attached to the address after the opcode.
  emit(P, Code, Op, Success);
  if (SI)
    SrcMap.emplace_back(Code.size(), SI);

  /// The initializer list forces the expression to be evaluated
  /// for each argument in the variadic template, in order.
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
