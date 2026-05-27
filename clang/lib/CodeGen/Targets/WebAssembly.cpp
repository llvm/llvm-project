//===- WebAssembly.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIInfoImpl.h"
#include "TargetInfo.h"

using namespace clang;
using namespace clang::CodeGen;

//===----------------------------------------------------------------------===//
// WebAssembly ABI Implementation
//
// This is a very simple ABI that relies a lot on DefaultABIInfo.
//===----------------------------------------------------------------------===//

class WebAssemblyABIInfo final : public ABIInfo {
  DefaultABIInfo defaultInfo;
  WebAssemblyABIKind Kind;

public:
  explicit WebAssemblyABIInfo(CodeGen::CodeGenTypes &CGT,
                              WebAssemblyABIKind Kind)
      : ABIInfo(CGT), defaultInfo(CGT), Kind(Kind) {}

private:
  ABIArgInfo classifyReturnType(QualType RetTy, llvm::CallingConv::ID CC) const;
  ABIArgInfo classifyArgumentType(QualType Ty, llvm::CallingConv::ID CC) const;

  // DefaultABIInfo's classifyReturnType and classifyArgumentType are
  // non-virtual, but computeInfo and EmitVAArg are virtual, so we
  // overload them.
  void computeInfo(CGFunctionInfo &FI) const override {
    llvm::CallingConv::ID CC = FI.getCallingConvention();
    if (!getCXXABI().classifyReturnType(FI))
      FI.getReturnInfo() = classifyReturnType(FI.getReturnType(), CC);
    for (auto &Arg : FI.arguments())
      Arg.info = classifyArgumentType(Arg.type, CC);
  }

  RValue EmitVAArg(CodeGenFunction &CGF, Address VAListAddr, QualType Ty,
                   AggValueSlot Slot) const override;
};

class WebAssemblyTargetCodeGenInfo final : public TargetCodeGenInfo {
public:
  explicit WebAssemblyTargetCodeGenInfo(CodeGen::CodeGenTypes &CGT,
                                        WebAssemblyABIKind K)
      : TargetCodeGenInfo(std::make_unique<WebAssemblyABIInfo>(CGT, K)) {
    SwiftInfo =
        std::make_unique<SwiftABIInfo>(CGT, /*SwiftErrorInRegister=*/false);
  }

  void setTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                           CodeGen::CodeGenModule &CGM) const override {
    TargetCodeGenInfo::setTargetAttributes(D, GV, CGM);
    if (const auto *FD = dyn_cast_or_null<FunctionDecl>(D)) {
      if (const auto *Attr = FD->getAttr<WebAssemblyImportModuleAttr>()) {
        llvm::Function *Fn = cast<llvm::Function>(GV);
        llvm::AttrBuilder B(GV->getContext());
        B.addAttribute("wasm-import-module", Attr->getImportModule());
        Fn->addFnAttrs(B);
      }
      if (const auto *Attr = FD->getAttr<WebAssemblyImportNameAttr>()) {
        llvm::Function *Fn = cast<llvm::Function>(GV);
        llvm::AttrBuilder B(GV->getContext());
        B.addAttribute("wasm-import-name", Attr->getImportName());
        Fn->addFnAttrs(B);
      }
      if (const auto *Attr = FD->getAttr<WebAssemblyExportNameAttr>()) {
        llvm::Function *Fn = cast<llvm::Function>(GV);
        llvm::AttrBuilder B(GV->getContext());
        B.addAttribute("wasm-export-name", Attr->getExportName());
        Fn->addFnAttrs(B);
      }
    }

    if (auto *FD = dyn_cast_or_null<FunctionDecl>(D)) {
      llvm::Function *Fn = cast<llvm::Function>(GV);
      if (!FD->doesThisDeclarationHaveABody() && !FD->hasPrototype())
        Fn->addFnAttr("no-prototype");
    }
  }

  /// Return the WebAssembly externref reference type.
  virtual llvm::Type *getWasmExternrefReferenceType() const override {
    return llvm::Type::getWasm_ExternrefTy(getABIInfo().getVMContext());
  }
  /// Return the WebAssembly funcref reference type.
  virtual llvm::Type *getWasmFuncrefReferenceType() const override {
    return llvm::Type::getWasm_FuncrefTy(getABIInfo().getVMContext());
  }
};

/// Count the number of "scalar fields" in the given record type, as defined by
/// WebAssembly/tool-conventions for the "wasm-multivalue" calling convention
/// primarily. A scalar field is a field that recursively, through nested
/// structs, unions, and arrays, contains just a single scalar value.
///
/// Returns the number of scalar fields, or std::nullopt if the record contains
/// a field that is not a scalar field (e.g., a sub-aggregate with multiple
/// scalars, a bit-field, or a flexible array member).
///
/// Note that this is similar to `isSingleElementStruct` in structure.
static std::optional<unsigned> countScalarFields(ASTContext &Context,
                                                 QualType T) {
  if (T->getAs<ComplexType>())
    return 2;
  const auto *RD = T->getAsRecordDecl();
  if (!RD)
    return std::nullopt;
  if (RD->hasFlexibleArrayMember())
    return std::nullopt;

  unsigned Count = 0;

  // Check bases first for C++ records.
  if (const auto *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
    for (const auto &Base : CXXRD->bases()) {
      auto SubCount = countScalarFields(Context, Base.getType());
      if (!SubCount)
        return std::nullopt;
      Count += *SubCount;
    }
  }

  for (const auto *FD : RD->fields()) {
    if (FD->isBitField())
      return std::nullopt;
    if (isEmptyField(Context, FD, true))
      continue;

    QualType T = FD->getType();
    if (isAggregateTypeForABI(T) && !isSingleElementStruct(T, Context))
      return std::nullopt;

    ++Count;
  }

  return Count;
}

/// Classify argument of given type \p Ty.
ABIArgInfo
WebAssemblyABIInfo::classifyArgumentType(QualType Ty,
                                         llvm::CallingConv::ID CC) const {
  Ty = useFirstFieldIfTransparentUnion(Ty);

  if (isAggregateTypeForABI(Ty)) {
    // Records with non-trivial destructors/copy-constructors should not be
    // passed by value.
    if (auto RAA = getRecordArgABI(Ty, getCXXABI()))
      return getNaturalAlignIndirect(Ty, getDataLayout().getAllocaAddrSpace(),
                                     RAA == CGCXXABI::RAA_DirectInMemory);
    // Ignore empty structs/unions.
    if (isEmptyRecord(getContext(), Ty, true))
      return ABIArgInfo::getIgnore();
    // Lower single-element structs to just pass a regular value. TODO: We
    // could do reasonable-size multiple-element structs too, using getExpand(),
    // though watch out for things like bitfields.
    if (const Type *SeltTy = isSingleElementStruct(Ty, getContext()))
      return ABIArgInfo::getDirect(CGT.ConvertType(QualType(SeltTy, 0)));
    // For the wasm-multivalue calling convention, structs with exactly two
    // scalar fields are passed directly as two arguments.
    if (CC == llvm::CallingConv::WASM_Multivalue) {
      if (auto N = countScalarFields(getContext(), Ty); N && *N == 2)
        return ABIArgInfo::getExpand();
    }
    // For the experimental multivalue ABI, fully expand all other aggregates
    if (Kind == WebAssemblyABIKind::ExperimentalMV) {
      if (Ty->getAs<ComplexType>())
        return ABIArgInfo::getDirect();
      const auto *RD = Ty->getAsRecordDecl();
      if (RD) {
        bool HasBitField = false;
        for (auto *Field : RD->fields()) {
          if (Field->isBitField()) {
            HasBitField = true;
            break;
          }
        }
        if (!HasBitField)
          return ABIArgInfo::getExpand();
      }
    }
  }

  // Otherwise just do the default thing.
  return defaultInfo.classifyArgumentType(Ty);
}

ABIArgInfo
WebAssemblyABIInfo::classifyReturnType(QualType RetTy,
                                       llvm::CallingConv::ID CC) const {
  if (isAggregateTypeForABI(RetTy)) {
    // Records with non-trivial destructors/copy-constructors should not be
    // returned by value.
    if (!getRecordArgABI(RetTy, getCXXABI())) {
      // Ignore empty structs/unions.
      if (isEmptyRecord(getContext(), RetTy, true))
        return ABIArgInfo::getIgnore();
      // Lower single-element structs to just return a regular value. TODO: We
      // could do reasonable-size multiple-element structs too, using
      // ABIArgInfo::getDirect().
      if (const Type *SeltTy = isSingleElementStruct(RetTy, getContext()))
        return ABIArgInfo::getDirect(CGT.ConvertType(QualType(SeltTy, 0)));
      // For the wasm-multivalue calling convention, structs whose fields are
      // (recursively) scalars are returned directly via the multivalue
      // proposal.
      if (CC == llvm::CallingConv::WASM_Multivalue) {
        if (auto N = countScalarFields(getContext(), RetTy);
            N && *N > 0 && *N <= 100)
          return ABIArgInfo::getDirect();
      }
      // For the experimental multivalue ABI, return all other aggregates
      if (Kind == WebAssemblyABIKind::ExperimentalMV)
        return ABIArgInfo::getDirect();
    }
  }

  // Otherwise just do the default thing.
  return defaultInfo.classifyReturnType(RetTy);
}

RValue WebAssemblyABIInfo::EmitVAArg(CodeGenFunction &CGF, Address VAListAddr,
                                     QualType Ty, AggValueSlot Slot) const {
  bool IsIndirect = isAggregateTypeForABI(Ty) &&
                    !isEmptyRecord(getContext(), Ty, true) &&
                    !isSingleElementStruct(Ty, getContext());
  return emitVoidPtrVAArg(CGF, VAListAddr, Ty, IsIndirect,
                          getContext().getTypeInfoInChars(Ty),
                          CharUnits::fromQuantity(4),
                          /*AllowHigherAlign=*/true, Slot);
}

std::unique_ptr<TargetCodeGenInfo>
CodeGen::createWebAssemblyTargetCodeGenInfo(CodeGenModule &CGM,
                                            WebAssemblyABIKind K) {
  return std::make_unique<WebAssemblyTargetCodeGenInfo>(CGM.getTypes(), K);
}
