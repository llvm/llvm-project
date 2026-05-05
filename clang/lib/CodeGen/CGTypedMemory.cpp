//===--- CGTypedMemory.cpp - Code Generation for Typed Memory Operations --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements code generation for typed memory operations.
//
//===----------------------------------------------------------------------===//

#include "CGCall.h"
#include "CGDebugInfo.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "ConstantEmitter.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypedMemoryDescriptorBits.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/APInt.h"
#include "llvm/CodeGen/MachineStableHash.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/xxhash.h"

using namespace clang;
using namespace CodeGen;

// The implementation is borrowed from SourceLocation::print but the file path
// is removed from the filename.
static void GetLocationHashTMO(llvm::raw_string_ostream &OS,
                               const SourceManager &SM, SourceLocation Loc) {
  if (!Loc.isValid()) {
    OS << "<invalid loc>";
    return;
  }

  if (Loc.isFileID()) {
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);

    if (PLoc.isInvalid()) {
      OS << "<invalid>";
      return;
    }
    // The macro expansion and spelling pos is identical for file locs.
    OS << llvm::sys::path::filename(PLoc.getFilename()) << ':' << PLoc.getLine()
       << ':' << PLoc.getColumn();
    return;
  }

  GetLocationHashTMO(OS, SM, SM.getExpansionLoc(Loc));

  OS << " <Spelling=";
  GetLocationHashTMO(OS, SM, SM.getSpellingLoc(Loc));
  OS << '>';
}

static llvm::stable_hash GetLocationHash(const CodeGenModule &CGM,
                                         SourceLocation Loc) {
  std::string LocationString;
  auto &SM = CGM.getContext().getSourceManager();
  llvm::raw_string_ostream OS(LocationString);
  GetLocationHashTMO(OS, SM, Loc);
  return llvm::xxh3_64bits(LocationString);
}

RValue CodeGenFunction::EmitTypedMemoryCall(const CallExpr *E,
                                            TypedMemoryAttr *TMA,
                                            ReturnValueSlot ReturnValue) {
  assert(CGM.getLangOpts().TypedMemoryOperations);
  auto *Target = TMA->getRewriteTarget();
  auto *TargetPrototype = cast<FunctionProtoType>(Target->getFunctionType());
  auto *OriginalDecl = E->getDirectCallee();
  auto *OriginalType = OriginalDecl->getType()->getAs<FunctionProtoType>();
  auto &Context = getContext();
  CGCallee Callee = EmitCallee(Target);
  auto PropertyDescriptorType = Context.getBitIntType(true, 64);
  CallArgList CallArgs;
  EmitCallArgs(CallArgs, OriginalType, E->arguments());

  auto InferredParamIndex = TMA->getInferredParameterIdx().getLLVMIndex();
  auto *InferredParameter = E->getArg(InferredParamIndex);

  TypedMemoryDescriptorBits Descriptor;
  if (auto InferredInfo = Context.getInferredInfoForCall(E);
      InferredInfo && InferredInfo->Type) {
    if (auto PrimaryType = InferredInfo->Type->primaryType()) {
      auto TypeDescriptor = Context.getTypedMemoryDescriptor(
          *PrimaryType, OO_None, InferredInfo->InferredCallsiteFlags);
      Descriptor = TypeDescriptor.asBits();
    }
  } else {
    auto LocationHash = GetLocationHash(CGM, E->getExprLoc());
    // We leave all descriptor flags as empty if we were unable to
    // infer the type for an expression, and use a hash of the call location
    // to allow callsite discrimination by the client.
    Descriptor.Hash = static_cast<uint32_t>(LocationHash);
  }
  Descriptor.Summary.TypeKind = TypedMemoryTypeKind::KindC;

  auto *DescriptorId = llvm::ConstantInt::get(CGM.Int64Ty, Descriptor.value());
  DescriptorId->setName("type_descriptor");
  auto *ConvertedValue = EmitScalarConversion(
      DescriptorId, PropertyDescriptorType,
      TargetPrototype->getParamType(InferredParamIndex + 1),
      InferredParameter->getExprLoc());
  auto InferredTypeArg =
      CallArg(RValue::get(ConvertedValue), PropertyDescriptorType);
  CallArgs.insert(CallArgs.begin() + InferredParamIndex + 1, InferredTypeArg);

  const CGFunctionInfo &FnInfo =
      CGM.getTypes().arrangeFreeFunctionCall(CallArgs, TargetPrototype, false);
  llvm::CallBase *CallOrInvoke = nullptr;
  RValue Call = EmitCall(FnInfo, Callee, ReturnValue, CallArgs, &CallOrInvoke,
                         /*IsMustTail=*/false, E->getExprLoc());

  // Add Annotation metadata to make the type descriptor available for tests.
  const std::string DescriptorValue = std::to_string(Descriptor.value());
  const std::string DescriptorHash = std::to_string(Descriptor.Hash);
  StringRef TypeSummaryDesc =
      Context.getTypeSummaryDescription(Descriptor.Summary);
  CallOrInvoke->addAnnotationMetadata(
      {"type-descriptor", DescriptorValue, DescriptorHash, TypeSummaryDesc});

  // Generate function declaration DISuprogram in order to be used
  // in debug info about call sites.
  if (CGDebugInfo *DI = getDebugInfo()) {
    if (auto *CalleeDecl = dyn_cast_or_null<FunctionDecl>(Target))
      DI->EmitFuncDeclForCallSite(CallOrInvoke, Target->getType(), CalleeDecl);
  }
  return Call;
}
