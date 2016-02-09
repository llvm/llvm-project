//==-- CGFunctionInfo.h - Representation of function argument/return types -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Defines CGFunctionInfo and associated types used in representing the
// LLVM source types and ABI-coerced types for function arguments and
// return values.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGEN_CGFUNCTIONINFO_H
#define LLVM_CLANG_CODEGEN_CGFUNCTIONINFO_H

#include "clang/AST/CanonicalType.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Type.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/TrailingObjects.h"
#include <cassert>

namespace llvm {
  class Type;
  class StructType;
}

namespace clang {
class Decl;

namespace CodeGen {

/// ABIArgInfo - Helper class to encapsulate information about how a
/// specific C type should be passed to or returned from a function.
class ABIArgInfo {
public:
  enum Kind : uint8_t {
    /// Direct - Pass the argument directly using the normal converted LLVM
    /// type, or by coercing to another specified type stored in
    /// 'CoerceToType').  If an offset is specified (in UIntData), then the
    /// argument passed is offset by some number of bytes in the memory
    /// representation. A dummy argument is emitted before the real argument
    /// if the specified type stored in "PaddingType" is not zero.
    Direct,

    /// Extend - Valid only for integer argument types. Same as 'direct'
    /// but also emit a zero/sign extension attribute.
    Extend,

    /// Indirect - Pass the argument indirectly via a hidden pointer
    /// with the specified alignment (0 indicates default alignment).
    Indirect,

    /// Ignore - Ignore the argument (treat as void). Useful for void and
    /// empty structs.
    Ignore,

    /// Expand - Only valid for aggregate argument types. The structure should
    /// be expanded into consecutive arguments for its constituent fields.
    /// Currently expand is only allowed on structures whose fields
    /// are all scalar types or are themselves expandable types.
    Expand,

    /// InAlloca - Pass the argument directly using the LLVM inalloca attribute.
    /// This is similar to indirect with byval, except it only applies to
    /// arguments stored in memory and forbids any implicit copies.  When
    /// applied to a return type, it means the value is returned indirectly via
    /// an implicit sret parameter stored in the argument struct.
    InAlloca,
    KindFirst = Direct,
    KindLast = InAlloca
  };

private:
  llvm::Type *TypeData; // isDirect() || isExtend()
  llvm::Type *PaddingType;
  union {
    unsigned DirectOffset;     // isDirect() || isExtend()
    unsigned IndirectAlign;    // isIndirect()
    unsigned AllocaFieldIndex; // isInAlloca()
  };
  Kind TheKind;
  bool PaddingInReg : 1;
  bool InAllocaSRet : 1;    // isInAlloca()
  bool IndirectByVal : 1;   // isIndirect()
  bool IndirectRealign : 1; // isIndirect()
  bool SRetAfterThis : 1;   // isIndirect()
  bool InReg : 1;           // isDirect() || isExtend() || isIndirect()
  bool CanBeFlattened: 1;   // isDirect()

  ABIArgInfo(Kind K)
      : PaddingType(nullptr), TheKind(K), PaddingInReg(false), InReg(false) {}

public:
  ABIArgInfo()
      : TypeData(nullptr), PaddingType(nullptr), DirectOffset(0),
        TheKind(Direct), PaddingInReg(false), InReg(false) {}

  static ABIArgInfo getDirect(llvm::Type *T = nullptr, unsigned Offset = 0,
                              llvm::Type *Padding = nullptr,
                              bool CanBeFlattened = true) {
    auto AI = ABIArgInfo(Direct);
    AI.setCoerceToType(T);
    AI.setDirectOffset(Offset);
    AI.setPaddingType(Padding);
    AI.setCanBeFlattened(CanBeFlattened);
    return AI;
  }
  static ABIArgInfo getDirectInReg(llvm::Type *T = nullptr) {
    auto AI = getDirect(T);
    AI.setInReg(true);
    return AI;
  }
  static ABIArgInfo getExtend(llvm::Type *T = nullptr) {
    auto AI = ABIArgInfo(Extend);
    AI.setCoerceToType(T);
    AI.setDirectOffset(0);
    return AI;
  }
  static ABIArgInfo getExtendInReg(llvm::Type *T = nullptr) {
    auto AI = getExtend(T);
    AI.setInReg(true);
    return AI;
  }
  static ABIArgInfo getIgnore() {
    return ABIArgInfo(Ignore);
  }
  static ABIArgInfo getIndirect(CharUnits Alignment, bool ByVal = true,
                                bool Realign = false,
                                llvm::Type *Padding = nullptr) {
    auto AI = ABIArgInfo(Indirect);
    AI.setIndirectAlign(Alignment);
    AI.setIndirectByVal(ByVal);
    AI.setIndirectRealign(Realign);
    AI.setSRetAfterThis(false);
    AI.setPaddingType(Padding);
    return AI;
  }
  static ABIArgInfo getIndirectInReg(CharUnits Alignment, bool ByVal = true,
                                     bool Realign = false) {
    auto AI = getIndirect(Alignment, ByVal, Realign);
    AI.setInReg(true);
    return AI;
  }
  static ABIArgInfo getInAlloca(unsigned FieldIndex) {
    auto AI = ABIArgInfo(InAlloca);
    AI.setInAllocaFieldIndex(FieldIndex);
    return AI;
  }
  static ABIArgInfo getExpand() {
    return ABIArgInfo(Expand);
  }
  static ABIArgInfo getExpandWithPadding(bool PaddingInReg,
                                         llvm::Type *Padding) {
    auto AI = getExpand();
    AI.setPaddingInReg(PaddingInReg);
    AI.setPaddingType(Padding);
    return AI;
  }

  Kind getKind() const { return TheKind; }
  bool isDirect() const { return TheKind == Direct; }
  bool isInAlloca() const { return TheKind == InAlloca; }
  bool isExtend() const { return TheKind == Extend; }
  bool isIgnore() const { return TheKind == Ignore; }
  bool isIndirect() const { return TheKind == Indirect; }
  bool isExpand() const { return TheKind == Expand; }

  bool canHaveCoerceToType() const { return isDirect() || isExtend(); }

  // Direct/Extend accessors
  unsigned getDirectOffset() const {
    assert((isDirect() || isExtend()) && "Not a direct or extend kind");
    return DirectOffset;
  }
  void setDirectOffset(unsigned Offset) {
    assert((isDirect() || isExtend()) && "Not a direct or extend kind");
    DirectOffset = Offset;
  }

  llvm::Type *getPaddingType() const { return PaddingType; }

  void setPaddingType(llvm::Type *T) { PaddingType = T; }

  bool getPaddingInReg() const {
    return PaddingInReg;
  }
  void setPaddingInReg(bool PIR) {
    PaddingInReg = PIR;
  }

  llvm::Type *getCoerceToType() const {
    assert(canHaveCoerceToType() && "Invalid kind!");
    return TypeData;
  }

  void setCoerceToType(llvm::Type *T) {
    assert(canHaveCoerceToType() && "Invalid kind!");
    TypeData = T;
  }

  bool getInReg() const {
    assert((isDirect() || isExtend() || isIndirect()) && "Invalid kind!");
    return InReg;
  }

  void setInReg(bool IR) {
    assert((isDirect() || isExtend() || isIndirect()) && "Invalid kind!");
    InReg = IR;
  }

  // Indirect accessors
  CharUnits getIndirectAlign() const {
    assert(isIndirect() && "Invalid kind!");
    return CharUnits::fromQuantity(IndirectAlign);
  }
  void setIndirectAlign(CharUnits IA) {
    assert(isIndirect() && "Invalid kind!");
    IndirectAlign = IA.getQuantity();
  }

  bool getIndirectByVal() const {
    assert(isIndirect() && "Invalid kind!");
    return IndirectByVal;
  }
  void setIndirectByVal(bool IBV) {
    assert(isIndirect() && "Invalid kind!");
    IndirectByVal = IBV;
  }

  bool getIndirectRealign() const {
    assert(isIndirect() && "Invalid kind!");
    return IndirectRealign;
  }
  void setIndirectRealign(bool IR) {
    assert(isIndirect() && "Invalid kind!");
    IndirectRealign = IR;
  }

  bool isSRetAfterThis() const {
    assert(isIndirect() && "Invalid kind!");
    return SRetAfterThis;
  }
  void setSRetAfterThis(bool AfterThis) {
    assert(isIndirect() && "Invalid kind!");
    SRetAfterThis = AfterThis;
  }

  unsigned getInAllocaFieldIndex() const {
    assert(isInAlloca() && "Invalid kind!");
    return AllocaFieldIndex;
  }
  void setInAllocaFieldIndex(unsigned FieldIndex) {
    assert(isInAlloca() && "Invalid kind!");
    AllocaFieldIndex = FieldIndex;
  }

  /// \brief Return true if this field of an inalloca struct should be returned
  /// to implement a struct return calling convention.
  bool getInAllocaSRet() const {
    assert(isInAlloca() && "Invalid kind!");
    return InAllocaSRet;
  }

  void setInAllocaSRet(bool SRet) {
    assert(isInAlloca() && "Invalid kind!");
    InAllocaSRet = SRet;
  }

  bool getCanBeFlattened() const {
    assert(isDirect() && "Invalid kind!");
    return CanBeFlattened;
  }

  void setCanBeFlattened(bool Flatten) {
    assert(isDirect() && "Invalid kind!");
    CanBeFlattened = Flatten;
  }

  void dump() const;
};

/// A class for recording the number of arguments that a function
/// signature requires.
class RequiredArgs {
  /// The number of required arguments, or ~0 if the signature does
  /// not permit optional arguments.
  unsigned NumRequired;
public:
  enum All_t { All };

  RequiredArgs(All_t _) : NumRequired(~0U) {}
  explicit RequiredArgs(unsigned n) : NumRequired(n) {
    assert(n != ~0U);
  }

  /// Compute the arguments required by the given formal prototype,
  /// given that there may be some additional, non-formal arguments
  /// in play.
  static RequiredArgs forPrototypePlus(const FunctionProtoType *prototype,
                                       unsigned additional) {
    if (!prototype->isVariadic()) return All;
    return RequiredArgs(prototype->getNumParams() + additional);
  }

  static RequiredArgs forPrototype(const FunctionProtoType *prototype) {
    return forPrototypePlus(prototype, 0);
  }

  static RequiredArgs forPrototype(CanQual<FunctionProtoType> prototype) {
    return forPrototype(prototype.getTypePtr());
  }

  static RequiredArgs forPrototypePlus(CanQual<FunctionProtoType> prototype,
                                       unsigned additional) {
    return forPrototypePlus(prototype.getTypePtr(), additional);
  }

  bool allowsOptionalArgs() const { return NumRequired != ~0U; }
  unsigned getNumRequiredArgs() const {
    assert(allowsOptionalArgs());
    return NumRequired;
  }

  unsigned getOpaqueData() const { return NumRequired; }
  static RequiredArgs getFromOpaqueData(unsigned value) {
    if (value == ~0U) return All;
    return RequiredArgs(value);
  }
};

// Implementation detail of CGFunctionInfo, factored out so it can be named
// in the TrailingObjects base class of CGFunctionInfo.
struct CGFunctionInfoArgInfo {
  CanQualType type;
  ABIArgInfo info;
};

/// CGFunctionInfo - Class to encapsulate the information about a
/// function definition.
class CGFunctionInfo final
    : public llvm::FoldingSetNode,
      private llvm::TrailingObjects<CGFunctionInfo, CGFunctionInfoArgInfo> {
  typedef CGFunctionInfoArgInfo ArgInfo;

  /// The LLVM::CallingConv to use for this function (as specified by the
  /// user).
  unsigned CallingConvention : 8;

  /// The LLVM::CallingConv to actually use for this function, which may
  /// depend on the ABI.
  unsigned EffectiveCallingConvention : 8;

  /// The clang::CallingConv that this was originally created with.
  unsigned ASTCallingConvention : 8;

  /// Whether this is an instance method.
  unsigned InstanceMethod : 1;

  /// Whether this is a chain call.
  unsigned ChainCall : 1;

  /// Whether this function is noreturn.
  unsigned NoReturn : 1;

  /// Whether this function is returns-retained.
  unsigned ReturnsRetained : 1;

  /// How many arguments to pass inreg.
  unsigned HasRegParm : 1;
  unsigned RegParm : 3;

  RequiredArgs Required;

  /// The struct representing all arguments passed in memory.  Only used when
  /// passing non-trivial types with inalloca.  Not part of the profile.
  llvm::StructType *ArgStruct;
  unsigned ArgStructAlign;

  unsigned NumArgs;

  ArgInfo *getArgsBuffer() {
    return getTrailingObjects<ArgInfo>();
  }
  const ArgInfo *getArgsBuffer() const {
    return getTrailingObjects<ArgInfo>();
  }

  size_t numTrailingObjects(OverloadToken<ArgInfo>) { return NumArgs + 1; }
  friend class TrailingObjects;

  CGFunctionInfo() : Required(RequiredArgs::All) {}

public:
  static CGFunctionInfo *create(unsigned llvmCC,
                                bool instanceMethod,
                                bool chainCall,
                                const FunctionType::ExtInfo &extInfo,
                                CanQualType resultType,
                                ArrayRef<CanQualType> argTypes,
                                RequiredArgs required);
  void operator delete(void *p) { TrailingObjects::operator delete(p); }

  typedef const ArgInfo *const_arg_iterator;
  typedef ArgInfo *arg_iterator;

  typedef llvm::iterator_range<arg_iterator> arg_range;
  typedef llvm::iterator_range<const_arg_iterator> arg_const_range;

  arg_range arguments() { return arg_range(arg_begin(), arg_end()); }
  arg_const_range arguments() const {
    return arg_const_range(arg_begin(), arg_end());
  }

  const_arg_iterator arg_begin() const { return getArgsBuffer() + 1; }
  const_arg_iterator arg_end() const { return getArgsBuffer() + 1 + NumArgs; }
  arg_iterator arg_begin() { return getArgsBuffer() + 1; }
  arg_iterator arg_end() { return getArgsBuffer() + 1 + NumArgs; }

  unsigned  arg_size() const { return NumArgs; }

  bool isVariadic() const { return Required.allowsOptionalArgs(); }
  RequiredArgs getRequiredArgs() const { return Required; }
  unsigned getNumRequiredArgs() const {
    return isVariadic() ? getRequiredArgs().getNumRequiredArgs() : arg_size();
  }

  bool isInstanceMethod() const { return InstanceMethod; }

  bool isChainCall() const { return ChainCall; }

  bool isNoReturn() const { return NoReturn; }

  /// In ARC, whether this function retains its return value.  This
  /// is not always reliable for call sites.
  bool isReturnsRetained() const { return ReturnsRetained; }

  /// getASTCallingConvention() - Return the AST-specified calling
  /// convention.
  CallingConv getASTCallingConvention() const {
    return CallingConv(ASTCallingConvention);
  }

  /// getCallingConvention - Return the user specified calling
  /// convention, which has been translated into an LLVM CC.
  unsigned getCallingConvention() const { return CallingConvention; }

  /// getEffectiveCallingConvention - Return the actual calling convention to
  /// use, which may depend on the ABI.
  unsigned getEffectiveCallingConvention() const {
    return EffectiveCallingConvention;
  }
  void setEffectiveCallingConvention(unsigned Value) {
    EffectiveCallingConvention = Value;
  }

  bool getHasRegParm() const { return HasRegParm; }
  unsigned getRegParm() const { return RegParm; }

  FunctionType::ExtInfo getExtInfo() const {
    return FunctionType::ExtInfo(isNoReturn(),
                                 getHasRegParm(), getRegParm(),
                                 getASTCallingConvention(),
                                 isReturnsRetained());
  }

  CanQualType getReturnType() const { return getArgsBuffer()[0].type; }

  ABIArgInfo &getReturnInfo() { return getArgsBuffer()[0].info; }
  const ABIArgInfo &getReturnInfo() const { return getArgsBuffer()[0].info; }

  /// \brief Return true if this function uses inalloca arguments.
  bool usesInAlloca() const { return ArgStruct; }

  /// \brief Get the struct type used to represent all the arguments in memory.
  llvm::StructType *getArgStruct() const { return ArgStruct; }
  CharUnits getArgStructAlignment() const {
    return CharUnits::fromQuantity(ArgStructAlign);
  }
  void setArgStruct(llvm::StructType *Ty, CharUnits Align) {
    ArgStruct = Ty;
    ArgStructAlign = Align.getQuantity();
  }

  void Profile(llvm::FoldingSetNodeID &ID) {
    ID.AddInteger(getASTCallingConvention());
    ID.AddBoolean(InstanceMethod);
    ID.AddBoolean(ChainCall);
    ID.AddBoolean(NoReturn);
    ID.AddBoolean(ReturnsRetained);
    ID.AddBoolean(HasRegParm);
    ID.AddInteger(RegParm);
    ID.AddInteger(Required.getOpaqueData());
    getReturnType().Profile(ID);
    for (const auto &I : arguments())
      I.type.Profile(ID);
  }
  static void Profile(llvm::FoldingSetNodeID &ID,
                      bool InstanceMethod,
                      bool ChainCall,
                      const FunctionType::ExtInfo &info,
                      RequiredArgs required,
                      CanQualType resultType,
                      ArrayRef<CanQualType> argTypes) {
    ID.AddInteger(info.getCC());
    ID.AddBoolean(InstanceMethod);
    ID.AddBoolean(ChainCall);
    ID.AddBoolean(info.getNoReturn());
    ID.AddBoolean(info.getProducesResult());
    ID.AddBoolean(info.getHasRegParm());
    ID.AddInteger(info.getRegParm());
    ID.AddInteger(required.getOpaqueData());
    resultType.Profile(ID);
    for (ArrayRef<CanQualType>::iterator
           i = argTypes.begin(), e = argTypes.end(); i != e; ++i) {
      i->Profile(ID);
    }
  }
};

/// CGCalleeInfo - Class to encapsulate the information about a callee to be
/// used during the generation of call/invoke instructions.
class CGCalleeInfo {
  /// \brief The function proto type of the callee.
  const FunctionProtoType *CalleeProtoTy;
  /// \brief The function declaration of the callee.
  const Decl *CalleeDecl;

public:
  explicit CGCalleeInfo() : CalleeProtoTy(nullptr), CalleeDecl(nullptr) {}
  CGCalleeInfo(const FunctionProtoType *calleeProtoTy, const Decl *calleeDecl)
      : CalleeProtoTy(calleeProtoTy), CalleeDecl(calleeDecl) {}
  CGCalleeInfo(const FunctionProtoType *calleeProtoTy)
      : CalleeProtoTy(calleeProtoTy), CalleeDecl(nullptr) {}
  CGCalleeInfo(const Decl *calleeDecl)
      : CalleeProtoTy(nullptr), CalleeDecl(calleeDecl) {}

  const FunctionProtoType *getCalleeFunctionProtoType() {
    return CalleeProtoTy;
  }
  const Decl *getCalleeDecl() { return CalleeDecl; }
};

}  // end namespace CodeGen
}  // end namespace clang

#endif
