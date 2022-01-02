//===-- Types.h - API Notes Data Types --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_APINOTES_TYPES_H
#define LLVM_CLANG_APINOTES_TYPES_H

#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include <climits>
#include <vector>

namespace clang {
namespace api_notes {
enum class RetainCountConventionKind {
  None,
  CFReturnsRetained,
  CFReturnsNotRetained,
  NSReturnsRetained,
  NSReturnsNotRetained,
};

/// The payload for an enum_extensibility attribute. This is a tri-state rather
/// than just a boolean because the presence of the attribute indicates
/// auditing.
enum class EnumExtensibilityKind {
  None,
  Open,
  Closed,
};

/// The kind of a swift_wrapper/swift_newtype.
enum class SwiftNewTypeKind {
  None,
  Struct,
  Enum,
};

/// Describes API notes data for any entity.
///
/// This is used as the base of all API notes.
class CommonEntityInfo {
public:
  /// Message to use when this entity is unavailable.
  std::string UnavailableMsg;

  /// Whether this entity is marked unavailable.
  unsigned Unavailable : 1;

  /// Whether this entity is marked unavailable in Swift.
  unsigned UnavailableInSwift : 1;

private:
  /// Whether SwiftPrivate was specified.
  unsigned SwiftPrivateSpecified : 1;

  /// Whether this entity is considered "private" to a Swift overlay.
  unsigned SwiftPrivate : 1;

public:
  /// Swift name of this entity.
  std::string SwiftName;

  CommonEntityInfo()
      : Unavailable(0), UnavailableInSwift(0), SwiftPrivateSpecified(0),
        SwiftPrivate(0) {}

  llvm::Optional<bool> isSwiftPrivate() const {
    return SwiftPrivateSpecified ? llvm::Optional<bool>(SwiftPrivate)
                                 : llvm::None;
  }

  void setSwiftPrivate(llvm::Optional<bool> Private) {
    SwiftPrivateSpecified = Private.hasValue();
    SwiftPrivate = Private.hasValue() ? *Private : 0;
  }

  friend bool operator==(const CommonEntityInfo &, const CommonEntityInfo &);

  CommonEntityInfo &operator|=(const CommonEntityInfo &RHS) {
    // Merge unavailability.
    if (RHS.Unavailable) {
      Unavailable = true;
      if (UnavailableMsg.empty())
        UnavailableMsg = RHS.UnavailableMsg;
    }

    if (RHS.UnavailableInSwift) {
      UnavailableInSwift = true;
      if (UnavailableMsg.empty())
        UnavailableMsg = RHS.UnavailableMsg;
    }

    if (!SwiftPrivateSpecified)
      setSwiftPrivate(RHS.isSwiftPrivate());

    if (SwiftName.empty())
      SwiftName = RHS.SwiftName;

    return *this;
  }

  LLVM_DUMP_METHOD void dump(llvm::raw_ostream &OS) const;
};

inline bool operator==(const CommonEntityInfo &LHS,
                       const CommonEntityInfo &RHS) {
  return LHS.UnavailableMsg == RHS.UnavailableMsg &&
         LHS.Unavailable == RHS.Unavailable &&
         LHS.UnavailableInSwift == RHS.UnavailableInSwift &&
         LHS.SwiftPrivateSpecified == RHS.SwiftPrivateSpecified &&
         LHS.SwiftPrivate == RHS.SwiftPrivate && LHS.SwiftName == RHS.SwiftName;
}

inline bool operator!=(const CommonEntityInfo &LHS,
                       const CommonEntityInfo &RHS) {
  return !(LHS == RHS);
}

/// Describes API notes for types.
class CommonTypeInfo : public CommonEntityInfo {
  /// The Swift type to which a given type is bridged.
  ///
  /// Reflects the swift_bridge attribute.
  llvm::Optional<std::string> SwiftBridge;

  /// The NS error domain for this type.
  llvm::Optional<std::string> NSErrorDomain;

public:
  CommonTypeInfo() {}

  const llvm::Optional<std::string> &getSwiftBridge() const {
    return SwiftBridge;
  }

  void setSwiftBridge(const llvm::Optional<std::string> &SwiftType) {
    SwiftBridge = SwiftType;
  }

  void setSwiftBridge(const llvm::Optional<llvm::StringRef> &SwiftType) {
    SwiftBridge = SwiftType
                      ? llvm::Optional<std::string>(std::string(*SwiftType))
                      : llvm::None;
  }

  const llvm::Optional<std::string> &getNSErrorDomain() const {
    return NSErrorDomain;
  }

  void setNSErrorDomain(const llvm::Optional<std::string> &Domain) {
    NSErrorDomain = Domain;
  }

  void setNSErrorDomain(const llvm::Optional<llvm::StringRef> &Domain) {
    NSErrorDomain =
        Domain ? llvm::Optional<std::string>(std::string(*Domain)) : llvm::None;
  }

  friend bool operator==(const CommonTypeInfo &, const CommonTypeInfo &);

  CommonTypeInfo &operator|=(const CommonTypeInfo &RHS) {
    // Merge inherited info.
    static_cast<CommonEntityInfo &>(*this) |= RHS;

    if (!SwiftBridge)
      setSwiftBridge(RHS.getSwiftBridge());
    if (!NSErrorDomain)
      setNSErrorDomain(RHS.getNSErrorDomain());

    return *this;
  }

  LLVM_DUMP_METHOD void dump(llvm::raw_ostream &OS) const;
};

inline bool operator==(const CommonTypeInfo &LHS, const CommonTypeInfo &RHS) {
  return static_cast<const CommonEntityInfo &>(LHS) == RHS &&
         LHS.SwiftBridge == RHS.SwiftBridge &&
         LHS.NSErrorDomain == RHS.NSErrorDomain;
}

inline bool operator!=(const CommonTypeInfo &LHS, const CommonTypeInfo &RHS) {
  return !(LHS == RHS);
}

/// Describes API notes data for an Objective-C class or protocol.
class ObjCContextInfo : public CommonTypeInfo {
  /// Whether this class has a default nullability.
  unsigned HasDefaultNullability : 1;

  /// The default nullability.
  unsigned DefaultNullability : 2;

  /// Whether this class has designated initializers recorded.
  unsigned HasDesignatedInits : 1;

  unsigned SwiftImportAsNonGenericSpecified : 1;
  unsigned SwiftImportAsNonGeneric : 1;

  unsigned SwiftObjCMembersSpecified : 1;
  unsigned SwiftObjCMembers : 1;

public:
  ObjCContextInfo()
      : HasDefaultNullability(0), DefaultNullability(0), HasDesignatedInits(0),
        SwiftImportAsNonGenericSpecified(false), SwiftImportAsNonGeneric(false),
        SwiftObjCMembersSpecified(false), SwiftObjCMembers(false) {}

  /// Determine the default nullability for properties and methods of this
  /// class.
  ///
  /// eturns the default nullability, if implied, or None if there is no
  llvm::Optional<NullabilityKind> getDefaultNullability() const {
    return HasDefaultNullability
               ? llvm::Optional<NullabilityKind>(
                     static_cast<NullabilityKind>(DefaultNullability))
               : llvm::None;
  }

  /// Set the default nullability for properties and methods of this class.
  void setDefaultNullability(NullabilityKind Kind) {
    HasDefaultNullability = true;
    DefaultNullability = static_cast<unsigned>(Kind);
  }

  bool hasDesignatedInits() const { return HasDesignatedInits; }
  void setHasDesignatedInits(bool Value) { HasDesignatedInits = Value; }

  llvm::Optional<bool> getSwiftImportAsNonGeneric() const {
    return SwiftImportAsNonGenericSpecified
               ? llvm::Optional<bool>(SwiftImportAsNonGeneric)
               : llvm::None;
  }
  void setSwiftImportAsNonGeneric(llvm::Optional<bool> Value) {
    SwiftImportAsNonGenericSpecified = Value.hasValue();
    SwiftImportAsNonGeneric = Value.getValueOr(false);
  }

  llvm::Optional<bool> getSwiftObjCMembers() const {
    return SwiftObjCMembersSpecified ? llvm::Optional<bool>(SwiftObjCMembers)
                                     : llvm::None;
  }
  void setSwiftObjCMembers(llvm::Optional<bool> Value) {
    SwiftObjCMembersSpecified = Value.hasValue();
    SwiftObjCMembers = Value.getValueOr(false);
  }

  /// Strip off any information within the class information structure that is
  /// module-local, such as 'audited' flags.
  void stripModuleLocalInfo() {
    HasDefaultNullability = false;
    DefaultNullability = 0;
  }

  friend bool operator==(const ObjCContextInfo &, const ObjCContextInfo &);

  ObjCContextInfo &operator|=(const ObjCContextInfo &RHS) {
    // Merge inherited info.
    static_cast<CommonTypeInfo &>(*this) |= RHS;

    // Merge nullability.
    if (!getDefaultNullability())
      if (auto Nullability = RHS.getDefaultNullability())
        setDefaultNullability(*Nullability);

    if (!SwiftImportAsNonGenericSpecified)
      setSwiftImportAsNonGeneric(RHS.getSwiftImportAsNonGeneric());

    if (!SwiftObjCMembersSpecified)
      setSwiftObjCMembers(RHS.getSwiftObjCMembers());

    HasDesignatedInits |= RHS.HasDesignatedInits;

    return *this;
  }

  LLVM_DUMP_METHOD void dump(llvm::raw_ostream &OS);
};

inline bool operator==(const ObjCContextInfo &LHS, const ObjCContextInfo &RHS) {
  return static_cast<const CommonTypeInfo &>(LHS) == RHS &&
         LHS.getDefaultNullability() == RHS.getDefaultNullability() &&
         LHS.HasDesignatedInits == RHS.HasDesignatedInits &&
         LHS.getSwiftImportAsNonGeneric() == RHS.getSwiftImportAsNonGeneric() &&
         LHS.getSwiftObjCMembers() == RHS.getSwiftObjCMembers();
}

inline bool operator!=(const ObjCContextInfo &LHS, const ObjCContextInfo &RHS) {
  return !(LHS == RHS);
}

/// API notes for a variable/property.
class VariableInfo : public CommonEntityInfo {
  /// Whether this property has been audited for nullability.
  unsigned NullabilityAudited : 1;

  /// The kind of nullability for this property. Only valid if the nullability
  /// has been audited.
  unsigned Nullable : 2;

  /// The C type of the variable, as a string.
  std::string Type;

public:
  VariableInfo() : NullabilityAudited(false), Nullable(0) {}

  llvm::Optional<NullabilityKind> getNullability() const {
    return NullabilityAudited ? llvm::Optional<NullabilityKind>(
                                    static_cast<NullabilityKind>(Nullable))
                              : llvm::None;
  }

  void setNullabilityAudited(NullabilityKind kind) {
    NullabilityAudited = true;
    Nullable = static_cast<unsigned>(kind);
  }

  const std::string &getType() const { return Type; }
  void setType(const std::string &type) { Type = type; }

  friend bool operator==(const VariableInfo &, const VariableInfo &);

  VariableInfo &operator|=(const VariableInfo &RHS) {
    static_cast<CommonEntityInfo &>(*this) |= RHS;

    if (!NullabilityAudited && RHS.NullabilityAudited)
      setNullabilityAudited(*RHS.getNullability());
    if (Type.empty())
      Type = RHS.Type;

    return *this;
  }

  LLVM_DUMP_METHOD void dump(llvm::raw_ostream &OS) const;
};

inline bool operator==(const VariableInfo &LHS, const VariableInfo &RHS) {
  return static_cast<const CommonEntityInfo &>(LHS) == RHS &&
         LHS.NullabilityAudited == RHS.NullabilityAudited &&
         LHS.Nullable == RHS.Nullable && LHS.Type == RHS.Type;
}

inline bool operator!=(const VariableInfo &LHS, const VariableInfo &RHS) {
  return !(LHS == RHS);
}

/// Describes API notes data for an Objective-C property.
class ObjCPropertyInfo : public VariableInfo {
  unsigned SwiftImportAsAccessorsSpecified : 1;
  unsigned SwiftImportAsAccessors : 1;

public:
  ObjCPropertyInfo()
      : SwiftImportAsAccessorsSpecified(false), SwiftImportAsAccessors(false) {}

  llvm::Optional<bool> getSwiftImportAsAccessors() const {
    return SwiftImportAsAccessorsSpecified
               ? llvm::Optional<bool>(SwiftImportAsAccessors)
               : llvm::None;
  }
  void setSwiftImportAsAccessors(llvm::Optional<bool> Value) {
    SwiftImportAsAccessorsSpecified = Value.hasValue();
    SwiftImportAsAccessors = Value.getValueOr(false);
  }

  friend bool operator==(const ObjCPropertyInfo &, const ObjCPropertyInfo &);

  /// Merge class-wide information into the given property.
  ObjCPropertyInfo &operator|=(const ObjCContextInfo &RHS) {
    static_cast<CommonEntityInfo &>(*this) |= RHS;

    // Merge nullability.
    if (!getNullability())
      if (auto Nullable = RHS.getDefaultNullability())
        setNullabilityAudited(*Nullable);

    return *this;
  }

  ObjCPropertyInfo &operator|=(const ObjCPropertyInfo &RHS) {
    static_cast<VariableInfo &>(*this) |= RHS;

    if (!SwiftImportAsAccessorsSpecified)
      setSwiftImportAsAccessors(RHS.getSwiftImportAsAccessors());

    return *this;
  }

  LLVM_DUMP_METHOD void dump(llvm::raw_ostream &OS) const;
};

inline bool operator==(const ObjCPropertyInfo &LHS,
                       const ObjCPropertyInfo &RHS) {
  return static_cast<const VariableInfo &>(LHS) == RHS &&
         LHS.getSwiftImportAsAccessors() == RHS.getSwiftImportAsAccessors();
}

inline bool operator!=(const ObjCPropertyInfo &LHS,
                       const ObjCPropertyInfo &RHS) {
  return !(LHS == RHS);
}

/// Describes a function or method parameter.
class ParamInfo : public VariableInfo {
  /// Whether noescape was specified.
  unsigned NoEscapeSpecified : 1;

  /// Whether the this parameter has the 'noescape' attribute.
  unsigned NoEscape : 1;

  /// A biased RetainCountConventionKind, where 0 means "unspecified".
  ///
  /// Only relevant for out-parameters.
  unsigned RawRetainCountConvention : 3;

public:
  ParamInfo()
      : NoEscapeSpecified(false), NoEscape(false), RawRetainCountConvention() {}

  llvm::Optional<bool> isNoEscape() const {
    if (!NoEscapeSpecified)
      return llvm::None;
    return NoEscape;
  }
  void setNoEscape(llvm::Optional<bool> Value) {
    NoEscapeSpecified = Value.hasValue();
    NoEscape = Value.getValueOr(false);
  }

  llvm::Optional<RetainCountConventionKind> getRetainCountConvention() const {
    if (!RawRetainCountConvention)
      return llvm::None;
    return static_cast<RetainCountConventionKind>(RawRetainCountConvention - 1);
  }
  void
  setRetainCountConvention(llvm::Optional<RetainCountConventionKind> Value) {
    RawRetainCountConvention =
        Value.hasValue() ? static_cast<unsigned>(Value.getValue()) + 1 : 0;
    assert(getRetainCountConvention() == Value && "bitfield too small");
  }

  ParamInfo &operator|=(const ParamInfo &RHS) {
    static_cast<VariableInfo &>(*this) |= RHS;

    if (!NoEscapeSpecified && RHS.NoEscapeSpecified) {
      NoEscapeSpecified = true;
      NoEscape = RHS.NoEscape;
    }

    if (!RawRetainCountConvention)
      RawRetainCountConvention = RHS.RawRetainCountConvention;

    return *this;
  }

  friend bool operator==(const ParamInfo &, const ParamInfo &);

  LLVM_DUMP_METHOD void dump(llvm::raw_ostream &OS) const;
};

inline bool operator==(const ParamInfo &LHS, const ParamInfo &RHS) {
  return static_cast<const VariableInfo &>(LHS) == RHS &&
         LHS.NoEscapeSpecified == RHS.NoEscapeSpecified &&
         LHS.NoEscape == RHS.NoEscape &&
         LHS.RawRetainCountConvention == RHS.RawRetainCountConvention;
}

inline bool operator!=(const ParamInfo &LHS, const ParamInfo &RHS) {
  return !(LHS == RHS);
}

/// API notes for a function or method.
class FunctionInfo : public CommonEntityInfo {
private:
  static constexpr const unsigned NullabilityKindMask = 0x3;
  static constexpr const unsigned NullabilityKindSize = 2;

  static constexpr const unsigned ReturnInfoIndex = 0;

public:
  // If yes, we consider all types to be non-nullable unless otherwise noted.
  // If this flag is not set, the pointer types are considered to have
  // unknown nullability.

  /// Whether the signature has been audited with respect to nullability.
  unsigned NullabilityAudited : 1;

  /// Number of types whose nullability is encoded with the NullabilityPayload.
  unsigned NumAdjustedNullable : 8;

  /// A biased RetainCountConventionKind, where 0 means "unspecified".
  unsigned RawRetainCountConvention : 3;

  // NullabilityKindSize bits are used to encode the nullability. The info
  // about the return type is stored at position 0, followed by the nullability
  // of the parameters.

  /// Stores the nullability of the return type and the parameters.
  uint64_t NullabilityPayload = 0;

  /// The result type of this function, as a C type.
  std::string ResultType;

  /// The function parameters.
  std::vector<ParamInfo> Params;

  FunctionInfo()
      : NullabilityAudited(false), NumAdjustedNullable(0),
        RawRetainCountConvention() {}

  static unsigned getMaxNullabilityIndex() {
    return ((sizeof(NullabilityPayload) * CHAR_BIT) / NullabilityKindSize);
  }

  void addTypeInfo(unsigned index, NullabilityKind kind) {
    assert(index <= getMaxNullabilityIndex());
    assert(static_cast<unsigned>(kind) < NullabilityKindMask);

    NullabilityAudited = true;
    if (NumAdjustedNullable < index + 1)
      NumAdjustedNullable = index + 1;

    // Mask the bits.
    NullabilityPayload &=
        ~(NullabilityKindMask << (index * NullabilityKindSize));

    // Set the value.
    unsigned kindValue = (static_cast<unsigned>(kind))
                         << (index * NullabilityKindSize);
    NullabilityPayload |= kindValue;
  }

  /// Adds the return type info.
  void addReturnTypeInfo(NullabilityKind kind) {
    addTypeInfo(ReturnInfoIndex, kind);
  }

  /// Adds the parameter type info.
  void addParamTypeInfo(unsigned index, NullabilityKind kind) {
    addTypeInfo(index + 1, kind);
  }

  NullabilityKind getParamTypeInfo(unsigned index) const {
    return getTypeInfo(index + 1);
  }

  NullabilityKind getReturnTypeInfo() const { return getTypeInfo(0); }

  llvm::Optional<RetainCountConventionKind> getRetainCountConvention() const {
    if (!RawRetainCountConvention)
      return llvm::None;
    return static_cast<RetainCountConventionKind>(RawRetainCountConvention - 1);
  }
  void
  setRetainCountConvention(llvm::Optional<RetainCountConventionKind> Value) {
    RawRetainCountConvention =
        Value.hasValue() ? static_cast<unsigned>(Value.getValue()) + 1 : 0;
    assert(getRetainCountConvention() == Value && "bitfield too small");
  }

  friend bool operator==(const FunctionInfo &, const FunctionInfo &);

private:
  NullabilityKind getTypeInfo(unsigned index) const {
    assert(NullabilityAudited &&
           "Checking the type adjustment on non-audited method.");

    // If we don't have info about this parameter, return the default.
    if (index > NumAdjustedNullable)
      return NullabilityKind::NonNull;
    auto nullability = NullabilityPayload >> (index * NullabilityKindSize);
    return static_cast<NullabilityKind>(nullability & NullabilityKindMask);
  }

public:
  LLVM_DUMP_METHOD void dump(llvm::raw_ostream &OS) const;
};

inline bool operator==(const FunctionInfo &LHS, const FunctionInfo &RHS) {
  return static_cast<const CommonEntityInfo &>(LHS) == RHS &&
         LHS.NullabilityAudited == RHS.NullabilityAudited &&
         LHS.NumAdjustedNullable == RHS.NumAdjustedNullable &&
         LHS.NullabilityPayload == RHS.NullabilityPayload &&
         LHS.ResultType == RHS.ResultType && LHS.Params == RHS.Params &&
         LHS.RawRetainCountConvention == RHS.RawRetainCountConvention;
}

inline bool operator!=(const FunctionInfo &LHS, const FunctionInfo &RHS) {
  return !(LHS == RHS);
}

/// Describes API notes data for an Objective-C method.
class ObjCMethodInfo : public FunctionInfo {
public:
  /// Whether this is a designated initializer of its class.
  unsigned DesignatedInit : 1;

  /// Whether this is a required initializer.
  unsigned RequiredInit : 1;

  ObjCMethodInfo() : DesignatedInit(false), RequiredInit(false) {}

  friend bool operator==(const ObjCMethodInfo &, const ObjCMethodInfo &);

  ObjCMethodInfo &operator|=(const ObjCContextInfo &RHS) {
    // Merge Nullability.
    if (!NullabilityAudited) {
      if (auto Nullable = RHS.getDefaultNullability()) {
        NullabilityAudited = true;
        addTypeInfo(0, *Nullable);
      }
    }
    return *this;
  }

  LLVM_DUMP_METHOD void dump(llvm::raw_ostream &OS);
};

inline bool operator==(const ObjCMethodInfo &LHS, const ObjCMethodInfo &RHS) {
  return static_cast<const FunctionInfo &>(LHS) == RHS &&
         LHS.DesignatedInit == RHS.DesignatedInit &&
         LHS.RequiredInit == RHS.RequiredInit;
}

inline bool operator!=(const ObjCMethodInfo &LHS, const ObjCMethodInfo &RHS) {
  return !(LHS == RHS);
}

/// Describes API notes data for a global variable.
class GlobalVariableInfo : public VariableInfo {
public:
  GlobalVariableInfo() {}
};

/// Describes API notes data for a global function.
class GlobalFunctionInfo : public FunctionInfo {
public:
  GlobalFunctionInfo() {}
};

/// Describes API notes data for an enumerator.
class EnumConstantInfo : public CommonEntityInfo {
public:
  EnumConstantInfo() {}
};

/// Describes API notes data for a tag.
class TagInfo : public CommonTypeInfo {
  unsigned HasFlagEnum : 1;
  unsigned IsFlagEnum : 1;

public:
  llvm::Optional<EnumExtensibilityKind> EnumExtensibility;

  TagInfo() : HasFlagEnum(0), IsFlagEnum(0) {}

  llvm::Optional<bool> isFlagEnum() const {
    if (HasFlagEnum)
      return IsFlagEnum;
    return llvm::None;
  }
  void setFlagEnum(llvm::Optional<bool> Value) {
    HasFlagEnum = Value.hasValue();
    IsFlagEnum = Value.getValueOr(false);
  }

  TagInfo &operator|=(const TagInfo &RHS) {
    static_cast<CommonTypeInfo &>(*this) |= RHS;

    if (!HasFlagEnum && HasFlagEnum)
      setFlagEnum(RHS.isFlagEnum());

    if (!EnumExtensibility.hasValue())
      EnumExtensibility = RHS.EnumExtensibility;

    return *this;
  }

  friend bool operator==(const TagInfo &, const TagInfo &);

  LLVM_DUMP_METHOD void dump(llvm::raw_ostream &OS);
};

inline bool operator==(const TagInfo &LHS, const TagInfo &RHS) {
  return static_cast<const CommonTypeInfo &>(LHS) == RHS &&
         LHS.isFlagEnum() == RHS.isFlagEnum() &&
         LHS.EnumExtensibility == RHS.EnumExtensibility;
}

inline bool operator!=(const TagInfo &LHS, const TagInfo &RHS) {
  return !(LHS == RHS);
}

/// Describes API notes data for a typedef.
class TypedefInfo : public CommonTypeInfo {
public:
  llvm::Optional<SwiftNewTypeKind> SwiftWrapper;

  TypedefInfo() {}

  TypedefInfo &operator|=(const TypedefInfo &RHS) {
    static_cast<CommonTypeInfo &>(*this) |= RHS;
    if (!SwiftWrapper.hasValue())
      SwiftWrapper = RHS.SwiftWrapper;
    return *this;
  }

  friend bool operator==(const TypedefInfo &, const TypedefInfo &);

  LLVM_DUMP_METHOD void dump(llvm::raw_ostream &OS) const;
};

inline bool operator==(const TypedefInfo &LHS, const TypedefInfo &RHS) {
  return static_cast<const CommonTypeInfo &>(LHS) == RHS &&
         LHS.SwiftWrapper == RHS.SwiftWrapper;
}

inline bool operator!=(const TypedefInfo &LHS, const TypedefInfo &RHS) {
  return !(LHS == RHS);
}
} // namespace api_notes
} // namespace clang

#endif
