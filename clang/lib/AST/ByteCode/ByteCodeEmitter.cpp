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
#include "IntegralAP.h"
#include "Opcode.h"
#include "Program.h"
#include "clang/AST/ASTLambda.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include <type_traits>

using namespace clang;
using namespace clang::interp;

void ByteCodeEmitter::compileFunc(const FunctionDecl *FuncDecl,
                                  Function *Func) {
  assert(FuncDecl);
  assert(Func);
  assert(FuncDecl->isThisDeclarationADefinition());

  // Manually created functions that haven't been assigned proper
  // parameters yet.
  if (!FuncDecl->param_empty() && !FuncDecl->param_begin())
    return;

  // Set up lambda captures.
  if (const auto *MD = dyn_cast<CXXMethodDecl>(FuncDecl);
      MD && isLambdaCallOperator(MD)) {
    // Set up lambda capture to closure record field mapping.
    const Record *R = P.getOrCreateRecord(MD->getParent());
    assert(R);
    llvm::DenseMap<const ValueDecl *, FieldDecl *> LC;
    FieldDecl *LTC;

    MD->getParent()->getCaptureFields(LC, LTC);

    for (auto Cap : LC) {
      unsigned Offset = R->getField(Cap.second)->Offset;
      this->LambdaCaptures[Cap.first] = {
          Offset, Cap.second->getType()->isReferenceType()};
    }
    if (LTC) {
      QualType CaptureType = R->getField(LTC)->Decl->getType();
      this->LambdaThisCapture = {R->getField(LTC)->Offset,
                                 CaptureType->isPointerOrReferenceType()};
    }
  }

  // Register parameters with their offset.
  unsigned ParamIndex = 0;
  unsigned Drop = Func->hasRVO() +
                  (Func->hasThisPointer() && !Func->isThisPointerExplicit());
  for (auto ParamOffset : llvm::drop_begin(Func->ParamOffsets, Drop)) {
    const ParmVarDecl *PD = FuncDecl->parameters()[ParamIndex];
    OptPrimType T = Ctx.classify(PD->getType());
    this->Params.insert({PD, {ParamOffset, T != std::nullopt}});
    ++ParamIndex;
  }

  Func->setDefined(true);

  // Lambda static invokers are a special case that we emit custom code for.
  bool IsEligibleForCompilation = Func->isLambdaStaticInvoker() ||
                                  FuncDecl->isConstexpr() ||
                                  FuncDecl->hasAttr<MSConstexprAttr>();

  // Compile the function body.
  if (!IsEligibleForCompilation || !visitFunc(FuncDecl)) {
    Func->setIsFullyCompiled(true);
    return;
  }

  // Create scopes from descriptors.
  llvm::SmallVector<Scope, 2> Scopes;
  for (auto &DS : Descriptors) {
    Scopes.emplace_back(std::move(DS));
  }

  // Set the function's code.
  Func->setCode(FuncDecl, NextLocalOffset, std::move(Code), std::move(SrcMap),
                std::move(Scopes), FuncDecl->hasBody());
  Func->setIsFullyCompiled(true);
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

  if (auto It = LabelRelocs.find(Label); It != LabelRelocs.end()) {
    for (unsigned Reloc : It->second) {
      using namespace llvm::support;

      // Rewrite the operand of all jumps to this label.
      void *Location = Code.data() + Reloc - align(sizeof(int32_t));
      assert(aligned(Location));
      const int32_t Offset = Target - static_cast<int64_t>(Reloc);
      endian::write<int32_t, llvm::endianness::native>(Location, Offset);
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
  if (auto It = LabelOffsets.find(Label); It != LabelOffsets.end())
    return It->second - Position;

  // Otherwise, record relocation and return dummy offset.
  LabelRelocs[Label].push_back(Position);
  return 0ull;
}

/// Helper to write bytecode and bail out if 32-bit offsets become invalid.
/// Pointers will be automatically marshalled as 32-bit IDs.
template <typename T>
static void emit(Program &P, llvm::SmallVectorImpl<std::byte> &Code,
                 const T &Val, bool &Success) {
  size_t ValPos = Code.size();
  size_t Size;

  if constexpr (std::is_pointer_v<T>)
    Size = align(sizeof(uint32_t));
  else
    Size = align(sizeof(T));

  if (ValPos + Size > std::numeric_limits<unsigned>::max()) {
    Success = false;
    return;
  }

  // Access must be aligned!
  assert(aligned(ValPos));
  assert(aligned(ValPos + Size));
  Code.resize_for_overwrite(ValPos + Size);

  if constexpr (!std::is_pointer_v<T>) {
    new (Code.data() + ValPos) T(Val);
  } else {
    uint32_t ID = P.getOrCreateNativePointer(Val);
    new (Code.data() + ValPos) uint32_t(ID);
  }
}

/// Emits a serializable value. These usually (potentially) contain
/// heap-allocated memory and aren't trivially copyable.
template <typename T>
static void emitSerialized(llvm::SmallVectorImpl<std::byte> &Code, const T &Val,
                           bool &Success) {
  size_t ValPos = Code.size();
  size_t Size = align(Val.bytesToSerialize());

  if (ValPos + Size > std::numeric_limits<unsigned>::max()) {
    Success = false;
    return;
  }

  // Access must be aligned!
  assert(aligned(ValPos));
  assert(aligned(ValPos + Size));
  Code.resize_for_overwrite(ValPos + Size);

  Val.serialize(Code.data() + ValPos);
}

template <>
void emit(Program &P, llvm::SmallVectorImpl<std::byte> &Code,
          const Floating &Val, bool &Success) {
  emitSerialized(Code, Val, Success);
}

template <>
void emit(Program &P, llvm::SmallVectorImpl<std::byte> &Code,
          const IntegralAP<false> &Val, bool &Success) {
  emitSerialized(Code, Val, Success);
}

template <>
void emit(Program &P, llvm::SmallVectorImpl<std::byte> &Code,
          const IntegralAP<true> &Val, bool &Success) {
  emitSerialized(Code, Val, Success);
}

template <>
void emit(Program &P, llvm::SmallVectorImpl<std::byte> &Code,
          const FixedPoint &Val, bool &Success) {
  emitSerialized(Code, Val, Success);
}

template <typename... Tys>
bool ByteCodeEmitter::emitOp(Opcode Op, const Tys &...Args,
                             const SourceInfo &SI) {
  bool Success = true;

  // The opcode is followed by arguments. The source info is
  // attached to the address after the opcode.
  emit(P, Code, Op, Success);
  if (LocOverride)
    SrcMap.emplace_back(Code.size(), *LocOverride);
  else if (SI)
    SrcMap.emplace_back(Code.size(), SI);

  (..., emit(P, Code, Args, Success));
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

bool ByteCodeEmitter::speculate(const CallExpr *E, const LabelTy &EndLabel) {
  const Expr *Arg = E->getArg(0);
  PrimType T = Ctx.classify(Arg->getType()).value_or(PT_Ptr);
  if (!this->emitBCP(getOffset(EndLabel), T, E))
    return false;
  if (!this->visit(Arg))
    return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Opcode emitters
//===----------------------------------------------------------------------===//

#define GET_LINK_IMPL
#include "Opcodes.inc"
#undef GET_LINK_IMPL
