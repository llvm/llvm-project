//==-- CIRGenFunctionInfo.h - Representation of fn argument/return types ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines CIRGenFunctionInfo and associated types used in representing the
// CIR source types and ABI-coerced types for function arguments and
// return values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_CIRGENFUNCTIONINFO_H
#define LLVM_CLANG_CIR_CIRGENFUNCTIONINFO_H

#include "clang/AST/CanonicalType.h"

#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/TrailingObjects.h"

#include "mlir/Dialect/CIR/IR/CIRTypes.h"

namespace cir {

/// ABIArgInfo - Helper class to encapsulate information about how a specific C
/// type should be passed to or returned from a function.
class ABIArgInfo {
public:
  enum Kind : uint8_t {
    /// Direct - Pass the argument directly using the normal converted CIR type,
    /// or by coercing to another specified type stored in 'CoerceToType'). If
    /// an offset is specified (in UIntData), then the argument passed is offset
    /// by some number of bytes in the memory representation. A dummy argument
    /// is emitted before the real argument if the specified type stored in
    /// "PaddingType" is not zero.
    Direct,

    /// Extend - Valid only for integer argument types. Same as 'direct' but
    /// also emit a zer/sign extension attribute.
    Extend,

    /// Indirect - Pass the argument indirectly via a hidden pointer with the
    /// specified alignment (0 indicates default alignment) and address space.
    Indirect,

    /// IndirectAliased - Similar to Indirect, but the pointer may be to an
    /// object that is otherwise referenced. The object is known to not be
    /// modified through any other references for the duration of the call, and
    /// the callee must not itself modify the object. Because C allows parameter
    /// variables to be modified and guarantees that they have unique addresses,
    /// the callee must defensively copy the object into a local variable if it
    /// might be modified or its address might be compared. Since those are
    /// uncommon, in principle this convention allows programs to avoid copies
    /// in more situations. However, it may introduce *extra* copies if the
    /// callee fails to prove that a copy is unnecessary and the caller
    /// naturally produces an unaliased object for the argument.
    IndirectAliased,

    /// Ignore - Ignore the argument (treat as void). Useful for void and empty
    /// structs.
    Ignore,

    /// Expand - Only valid for aggregate argument types. The structure should
    /// be expanded into consecutive arguments for its constituent fields.
    /// Currently expand is only allowed on structures whose fields are all
    /// scalar types or are themselves expandable types.
    Expand,

    /// CoerceAndExpand - Only valid for aggregate argument types. The structure
    /// should be expanded into consecutive arguments corresponding to the
    /// non-array elements of the type stored in CoerceToType.
    /// Array elements in the type are assumed to be padding and skipped.
    CoerceAndExpand,

    // TODO: translate this idea to CIR! Define it for now just to ensure that
    // we can assert it not being used
    InAlloca,
    KindFirst = Direct,
    KindLast = InAlloca
  };

private:
  mlir::Type TypeData; // canHaveCoerceToType();
  union {
    mlir::Type PaddingType;                 // canHavePaddingType()
    mlir::Type UnpaddedCoerceAndExpandType; // isCoerceAndExpand()
  };
  struct DirectAttrInfo {
    unsigned Offset;
    unsigned Align;
  };
  struct IndirectAttrInfo {
    unsigned Align;
    unsigned AddrSpace;
  };
  union {
    DirectAttrInfo DirectAttr;     // isDirect() || isExtend()
    IndirectAttrInfo IndirectAttr; // isIndirect()
    unsigned AllocaFieldIndex;     // isInAlloca()
  };
  Kind TheKind;
  bool CanBeFlattened : 1; // isDirect()

  bool canHavePaddingType() const {
    return isDirect() || isExtend() || isIndirect() || isIndirectAliased() ||
           isExpand();
  }

  void setPaddingType(mlir::Type T) {
    assert(canHavePaddingType());
    PaddingType = T;
  }

public:
  ABIArgInfo(Kind K = Direct)
      : TypeData(nullptr), PaddingType(nullptr), DirectAttr{0, 0}, TheKind(K),
        CanBeFlattened(false) {}

  static ABIArgInfo getDirect(mlir::Type T = nullptr, unsigned Offset = 0,
                              mlir::Type Padding = nullptr,
                              bool CanBeFlattened = true, unsigned Align = 0) {
    auto AI = ABIArgInfo(Direct);
    AI.setCoerceToType(T);
    AI.setPaddingType(Padding);
    AI.setDirectOffset(Offset);
    AI.setDirectAlign(Align);
    AI.setCanBeFlattened(CanBeFlattened);
    return AI;
  }

  static ABIArgInfo getIgnore() { return ABIArgInfo(Ignore); }

  Kind getKind() const { return TheKind; }
  bool isDirect() const { return TheKind == Direct; }
  bool isInAlloca() const { return TheKind == InAlloca; }
  bool isExtend() const { return TheKind == Extend; }
  bool isIndirect() const { return TheKind == Indirect; }
  bool isIndirectAliased() const { return TheKind == IndirectAliased; }
  bool isExpand() const { return TheKind == Expand; }
  bool isCoerceAndExpand() const { return TheKind == CoerceAndExpand; }

  bool canHaveCoerceToType() const {
    return isDirect() || isExtend() || isCoerceAndExpand();
  }

  // Direct/Extend accessors
  unsigned getDirectOffset() const {
    assert((isDirect() || isExtend()) && "Not a direct or extend kind");
    return DirectAttr.Offset;
  }

  void setDirectOffset(unsigned Offset) {
    assert((isDirect() || isExtend()) && "Not a direct or extend kind");
    DirectAttr.Offset = Offset;
  }

  void setDirectAlign(unsigned Align) {
    assert((isDirect() || isExtend()) && "Not a direct or extend kind");
    DirectAttr.Align = Align;
  }

  void setCanBeFlattened(bool Flatten) {
    assert(isDirect() && "Invalid kind!");
    CanBeFlattened = Flatten;
  }

  mlir::Type getPaddingType() const {
    return (canHavePaddingType() ? PaddingType : nullptr);
  }

  mlir::Type getCoerceToType() const {
    assert(canHaveCoerceToType() && "Invalid kind!");
    return TypeData;
  }

  void setCoerceToType(mlir::Type T) {
    assert(canHaveCoerceToType() && "Invalid kind!");
    TypeData = T;
  }
};

struct CIRGenFunctionInfoArgInfo {
  clang::CanQualType type;
  ABIArgInfo info;
};

/// A class for recording the number of arguments that a function signature
/// requires.
class RequiredArgs {
  /// The number of required arguments, or ~0 if the signature does not permit
  /// optional arguments.
  unsigned NumRequired;

public:
  enum All_t { All };

  RequiredArgs(All_t _) : NumRequired(~0U) {}
  explicit RequiredArgs(unsigned n) : NumRequired(n) { assert(n != ~0U); }

  unsigned getOpaqueData() const { return NumRequired; }

  bool allowsOptionalArgs() const { return NumRequired != ~0U; }

  /// Compute the arguments required by the given formal prototype, given that
  /// there may be some additional, non-formal arguments in play.
  ///
  /// If FD is not null, this will consider pass_object_size params in FD.
  static RequiredArgs
  forPrototypePlus(const clang::FunctionProtoType *prototype,
                   unsigned additional) {
    assert(!prototype->isVariadic() && "NYI");
    return All;
  }

  static RequiredArgs
  forPrototypePlus(clang::CanQual<clang::FunctionProtoType> prototype,
                   unsigned additional) {
    return forPrototypePlus(prototype.getTypePtr(), additional);
  }

  unsigned getNumRequiredArgs() const {
    assert(allowsOptionalArgs());
    return NumRequired;
  }
};

class CIRGenFunctionInfo final
    : public llvm::FoldingSetNode,
      private llvm::TrailingObjects<
          CIRGenFunctionInfo, CIRGenFunctionInfoArgInfo,
          clang::FunctionProtoType::ExtParameterInfo> {

  typedef CIRGenFunctionInfoArgInfo ArgInfo;
  typedef clang::FunctionProtoType::ExtParameterInfo ExtParameterInfo;

  /// The cir::CallingConv to use for this function (as specified by the user).
  unsigned CallingConvention : 8;

  /// The cir::CallingConv to actually use for this function, which may depend
  /// on the ABI.
  unsigned EffectiveCallingConvention : 8;

  /// The clang::CallingConv that this was originally created with.
  unsigned ASTCallingConvention : 6;

  /// Whether this is an instance method.
  unsigned InstanceMethod : 1;

  /// Whether this is a chain call.
  unsigned ChainCall : 1;

  /// Whether this function is a CMSE nonsecure call
  unsigned CmseNSCall : 1;

  /// Whether this function is noreturn.
  unsigned NoReturn : 1;

  /// Whether this function is returns-retained.
  unsigned ReturnsRetained : 1;

  /// Whether this function saved caller registers.
  unsigned NoCallerSavedRegs : 1;

  /// How many arguments to pass inreg.
  unsigned HasRegParm : 1;
  unsigned RegParm : 3;

  /// Whether this function has nocf_check attribute.
  unsigned NoCfCheck : 1;

  RequiredArgs Required;

  /// The struct representing all arguments passed in memory. Only used when
  /// passing non-trivial types with inalloca. Not part of the profile.
  /// TODO: think about modeling this properly, this is just a dumb subsitution
  /// for now since we arent supporting anything other than arguments in
  /// registers atm
  mlir::cir::StructType *ArgStruct;
  unsigned ArgStructAlign : 31;
  unsigned HasExtParameterInfos : 1;

  unsigned NumArgs;

  ArgInfo *getArgsBuffer() { return getTrailingObjects<ArgInfo>(); }

  const ArgInfo *getArgsBuffer() const { return getTrailingObjects<ArgInfo>(); }

  ExtParameterInfo *getExtParameterInfosBuffer() {
    return getTrailingObjects<ExtParameterInfo>();
  }

  const ExtParameterInfo *getExtParameterInfosBuffer() const {
    return getTrailingObjects<ExtParameterInfo>();
  }

  CIRGenFunctionInfo() : Required(RequiredArgs::All) {}

public:
  static CIRGenFunctionInfo *create(unsigned cirCC, bool instanceMethod,
                                    bool chainCall,
                                    const clang::FunctionType::ExtInfo &extInfo,
                                    llvm::ArrayRef<ExtParameterInfo> paramInfos,
                                    clang::CanQualType resultType,
                                    llvm::ArrayRef<clang::CanQualType> argTypes,
                                    RequiredArgs required);
  void operator delete(void *p) { ::operator delete(p); }

  // Friending class TrailingObjects is apparantly not good enough for MSVC, so
  // these have to be public.
  friend class TrailingObjects;
  size_t numTrailingObjects(OverloadToken<ArgInfo>) const {
    return NumArgs + 1;
  }
  size_t numTrailingObjects(OverloadToken<ExtParameterInfo>) const {
    return (HasExtParameterInfos ? NumArgs : 0);
  }

  using const_arg_iterator = const ArgInfo *;
  using arg_iterator = ArgInfo *;

  static void Profile(llvm::FoldingSetNodeID &ID, bool InstanceMethod,
                      bool ChainCall, const clang::FunctionType::ExtInfo &info,
                      llvm::ArrayRef<ExtParameterInfo> paramInfos,
                      RequiredArgs required, clang::CanQualType resultType,
                      llvm::ArrayRef<clang::CanQualType> argTypes) {
    ID.AddInteger(info.getCC());
    ID.AddBoolean(InstanceMethod);
    ID.AddBoolean(info.getNoReturn());
    ID.AddBoolean(info.getProducesResult());
    ID.AddBoolean(info.getNoCallerSavedRegs());
    ID.AddBoolean(info.getHasRegParm());
    ID.AddBoolean(info.getRegParm());
    ID.AddBoolean(info.getNoCfCheck());
    ID.AddBoolean(info.getCmseNSCall());
    ID.AddBoolean(required.getOpaqueData());
    ID.AddBoolean(!paramInfos.empty());
    if (!paramInfos.empty()) {
      for (auto paramInfo : paramInfos)
        ID.AddInteger(paramInfo.getOpaqueValue());
    }
    resultType.Profile(ID);
    for (auto i : argTypes)
      i.Profile(ID);
  }

  /// getASTCallingConvention() - Return the AST-specified calling convention
  clang::CallingConv getASTCallingConvention() const {
    return clang::CallingConv(ASTCallingConvention);
  }

  void Profile(llvm::FoldingSetNodeID &ID) {
    ID.AddInteger(getASTCallingConvention());
    ID.AddBoolean(InstanceMethod);
    ID.AddBoolean(ChainCall);
    ID.AddBoolean(NoReturn);
    ID.AddBoolean(ReturnsRetained);
    ID.AddBoolean(NoCallerSavedRegs);
    ID.AddBoolean(HasRegParm);
    ID.AddBoolean(RegParm);
    ID.AddBoolean(NoCfCheck);
    ID.AddBoolean(CmseNSCall);
    ID.AddInteger(Required.getOpaqueData());
    ID.AddBoolean(HasExtParameterInfos);
    if (HasExtParameterInfos) {
      for (auto paramInfo : getExtParameterInfos())
        ID.AddInteger(paramInfo.getOpaqueValue());
    }
    getReturnType().Profile(ID);
    for (const auto &I : arguments())
      I.type.Profile(ID);
  }

  llvm::MutableArrayRef<ArgInfo> arguments() {
    return llvm::MutableArrayRef<ArgInfo>(arg_begin(), NumArgs);
  }
  llvm::ArrayRef<ArgInfo> arguments() const {
    return llvm::ArrayRef<ArgInfo>(arg_begin(), NumArgs);
  }

  const_arg_iterator arg_begin() const { return getArgsBuffer() + 1; }
  const_arg_iterator arg_end() const { return getArgsBuffer() + 1 + NumArgs; }
  arg_iterator arg_begin() { return getArgsBuffer() + 1; }
  arg_iterator arg_end() { return getArgsBuffer() + 1 + NumArgs; }

  unsigned arg_size() const { return NumArgs; }

  llvm::ArrayRef<ExtParameterInfo> getExtParameterInfos() const {
    if (!HasExtParameterInfos)
      return {};
    return llvm::ArrayRef(getExtParameterInfosBuffer(), NumArgs);
  }
  ExtParameterInfo getExtParameterInfo(unsigned argIndex) const {
    assert(argIndex <= NumArgs);
    if (!HasExtParameterInfos)
      return ExtParameterInfo();
    return getExtParameterInfos()[argIndex];
  }

  /// getCallingConvention - REturn the user specified calling convention, which
  /// has been translated into a CIR CC.
  unsigned getCallingConvention() const { return CallingConvention; }

  clang::CanQualType getReturnType() const { return getArgsBuffer()[0].type; }

  ABIArgInfo &getReturnInfo() { return getArgsBuffer()[0].info; }
  const ABIArgInfo &getReturnInfo() const { return getArgsBuffer()[0].info; }

  bool isChainCall() const { return ChainCall; }

  bool isVariadic() const { return Required.allowsOptionalArgs(); }
  RequiredArgs getRequiredArgs() const { return Required; }
  unsigned getNumRequiredArgs() const {
    assert(!isVariadic() && "Variadic NYI");
    return isVariadic() ? getRequiredArgs().getNumRequiredArgs() : arg_size();
  }

  mlir::cir::StructType *getArgStruct() const { return ArgStruct; }

  /// Return true if this function uses inalloca arguments.
  bool usesInAlloca() const { return ArgStruct; }
};

} // namespace cir

#endif
