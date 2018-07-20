//===--- Types.h - API Notes Data Types --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines data types used in the representation of API notes data.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_API_NOTES_TYPES_H
#define LLVM_CLANG_API_NOTES_TYPES_H
#include "clang/Basic/LLVM.h"
#include "clang/Basic/Specifiers.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include <cassert>
#include <climits>

namespace clang {
namespace api_notes {

/// The file extension used for the source representation of API notes.
static const char SOURCE_APINOTES_EXTENSION[] = "apinotes";

/// The file extension used for the binary representation of API notes.
static const char BINARY_APINOTES_EXTENSION[] = "apinotesc";

/// Opaque context ID used to refer to an Objective-C class or protocol.
class ContextID {
public:
  unsigned Value;

  explicit ContextID(unsigned value) : Value(value) { }
};

enum class RetainCountConventionKind {
  None,
  CFReturnsRetained,
  CFReturnsNotRetained,
  NSReturnsRetained,
  NSReturnsNotRetained,
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
      SwiftPrivate(0) { }

  Optional<bool> isSwiftPrivate() const {
    if (!SwiftPrivateSpecified) return None;
    return SwiftPrivate;
  }

  void setSwiftPrivate(Optional<bool> swiftPrivate) {
    if (swiftPrivate) {
      SwiftPrivateSpecified = 1;
      SwiftPrivate = *swiftPrivate;
    } else {
      SwiftPrivateSpecified = 0;
      SwiftPrivate = 0;
    }
  }

  friend bool operator==(const CommonEntityInfo &lhs,
                         const CommonEntityInfo &rhs) {
    return lhs.UnavailableMsg == rhs.UnavailableMsg &&
           lhs.Unavailable == rhs.Unavailable &&
           lhs.UnavailableInSwift == rhs.UnavailableInSwift &&
           lhs.SwiftPrivateSpecified == rhs.SwiftPrivateSpecified &&
           lhs.SwiftPrivate == rhs.SwiftPrivate &&
           lhs.SwiftName == rhs.SwiftName;
  }

  friend bool operator!=(const CommonEntityInfo &lhs,
                         const CommonEntityInfo &rhs) {
    return !(lhs == rhs);
  }

  friend CommonEntityInfo &operator|=(CommonEntityInfo &lhs,
                                      const CommonEntityInfo &rhs) {
    // Merge unavailability.
    if (rhs.Unavailable) {
      lhs.Unavailable = true;
      if (rhs.UnavailableMsg.length() != 0 &&
          lhs.UnavailableMsg.length() == 0) {
        lhs.UnavailableMsg = rhs.UnavailableMsg;
      }
    }

    if (rhs.UnavailableInSwift) {
      lhs.UnavailableInSwift = true;
      if (rhs.UnavailableMsg.length() != 0 &&
          lhs.UnavailableMsg.length() == 0) {
        lhs.UnavailableMsg = rhs.UnavailableMsg;
      }
    }

    if (rhs.SwiftPrivateSpecified && !lhs.SwiftPrivateSpecified) {
      lhs.SwiftPrivateSpecified = 1;
      lhs.SwiftPrivate = rhs.SwiftPrivate;
    }

    if (rhs.SwiftName.length() != 0 &&
        lhs.SwiftName.length() == 0)
      lhs.SwiftName = rhs.SwiftName;

    return lhs;
  }
};

/// Describes API notes for types.
class CommonTypeInfo : public CommonEntityInfo {
  /// The Swift type to which a given type is bridged.
  ///
  /// Reflects the swift_bridge attribute.
  Optional<std::string> SwiftBridge;

  /// The NS error domain for this type.
  Optional<std::string> NSErrorDomain;

public:
  CommonTypeInfo() : CommonEntityInfo() { }

  const Optional<std::string> &getSwiftBridge() const { return SwiftBridge; }

  void setSwiftBridge(const Optional<std::string> &swiftType) {
    SwiftBridge = swiftType;
  }

  void setSwiftBridge(const Optional<StringRef> &swiftType) {
    if (swiftType)
      SwiftBridge = *swiftType;
    else
      SwiftBridge = None;
  }

  const Optional<std::string> &getNSErrorDomain() const {
    return NSErrorDomain;
  }

  void setNSErrorDomain(const Optional<std::string> &domain) {
    NSErrorDomain = domain;
  }

  void setNSErrorDomain(const Optional<StringRef> &domain) {
    if (domain)
      NSErrorDomain = *domain;
    else
      NSErrorDomain = None;
  }

  friend CommonTypeInfo &operator|=(CommonTypeInfo &lhs,
                                    const CommonTypeInfo &rhs) {
    static_cast<CommonEntityInfo &>(lhs) |= rhs;
    if (!lhs.SwiftBridge && rhs.SwiftBridge)
      lhs.SwiftBridge = rhs.SwiftBridge;
    if (!lhs.NSErrorDomain && rhs.NSErrorDomain)
      lhs.NSErrorDomain = rhs.NSErrorDomain;
    return lhs;
  }

  friend bool operator==(const CommonTypeInfo &lhs,
                         const CommonTypeInfo &rhs) {
    return static_cast<const CommonEntityInfo &>(lhs) == rhs &&
      lhs.SwiftBridge == rhs.SwiftBridge &&
      lhs.NSErrorDomain == rhs.NSErrorDomain;
  }

  friend bool operator!=(const CommonTypeInfo &lhs,
                         const CommonTypeInfo &rhs) {
    return !(lhs == rhs);
  }
};

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
    : CommonTypeInfo(),
      HasDefaultNullability(0),
      DefaultNullability(0),
      HasDesignatedInits(0),
      SwiftImportAsNonGenericSpecified(false),
      SwiftImportAsNonGeneric(false),
      SwiftObjCMembersSpecified(false),
      SwiftObjCMembers(false)
  { }

  /// Determine the default nullability for properties and methods of this
  /// class.
  ///
  /// \returns the default nullability, if implied, or None if there is no
  Optional<NullabilityKind> getDefaultNullability() const {
    if (HasDefaultNullability)
      return static_cast<NullabilityKind>(DefaultNullability);

    return None;
  }

  /// Set the default nullability for properties and methods of this class.
  void setDefaultNullability(NullabilityKind kind) {
    HasDefaultNullability = true;
    DefaultNullability = static_cast<unsigned>(kind);
  }

  bool hasDesignatedInits() const { return HasDesignatedInits; }
  void setHasDesignatedInits(bool value) { HasDesignatedInits = value; }

  Optional<bool> getSwiftImportAsNonGeneric() const {
    if (SwiftImportAsNonGenericSpecified)
      return SwiftImportAsNonGeneric;
    return None;
  }
  void setSwiftImportAsNonGeneric(Optional<bool> value) {
    if (value.hasValue()) {
      SwiftImportAsNonGenericSpecified = true;
      SwiftImportAsNonGeneric = value.getValue();
    } else {
      SwiftImportAsNonGenericSpecified = false;
      SwiftImportAsNonGeneric = false;
    }
  }

  Optional<bool> getSwiftObjCMembers() const {
    if (SwiftObjCMembersSpecified)
      return SwiftObjCMembers;
    return None;
  }
  void setSwiftObjCMembers(Optional<bool> value) {
    SwiftObjCMembersSpecified = value.hasValue();
    SwiftObjCMembers = value.hasValue() ? *value : false;
  }

  /// Strip off any information within the class information structure that is
  /// module-local, such as 'audited' flags.
  void stripModuleLocalInfo() {
    HasDefaultNullability = false;
    DefaultNullability = 0;
  }

  friend bool operator==(const ObjCContextInfo &lhs, const ObjCContextInfo &rhs) {
    return static_cast<const CommonTypeInfo &>(lhs) == rhs &&
           lhs.getDefaultNullability() == rhs.getDefaultNullability() &&
           lhs.HasDesignatedInits == rhs.HasDesignatedInits &&
           lhs.getSwiftImportAsNonGeneric() ==
             rhs.getSwiftImportAsNonGeneric() &&
           lhs.getSwiftObjCMembers() == rhs.getSwiftObjCMembers();
  }

  friend bool operator!=(const ObjCContextInfo &lhs, const ObjCContextInfo &rhs) {
    return !(lhs == rhs);
  }

  friend ObjCContextInfo &operator|=(ObjCContextInfo &lhs,
                                     const ObjCContextInfo &rhs) {
    // Merge inherited info.
    static_cast<CommonTypeInfo &>(lhs) |= rhs;

    // Merge nullability.
    if (!lhs.getDefaultNullability()) {
      if (auto nullable = rhs.getDefaultNullability()) {
        lhs.setDefaultNullability(*nullable);
      }
    }

    if (!lhs.SwiftImportAsNonGenericSpecified &&
        rhs.SwiftImportAsNonGenericSpecified) {
      lhs.SwiftImportAsNonGenericSpecified = true;
      lhs.SwiftImportAsNonGeneric = rhs.SwiftImportAsNonGeneric;
    }

    if (!lhs.SwiftObjCMembersSpecified && rhs.SwiftObjCMembersSpecified) {
      lhs.SwiftObjCMembersSpecified = true;
      lhs.SwiftObjCMembers = rhs.SwiftObjCMembers;
    }

    lhs.HasDesignatedInits |= rhs.HasDesignatedInits;

    return lhs;
  }
  
  void dump(llvm::raw_ostream &os);
};

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
  VariableInfo()
    : CommonEntityInfo(),
      NullabilityAudited(false),
      Nullable(0) { }

  Optional<NullabilityKind> getNullability() const {
    if (NullabilityAudited)
      return static_cast<NullabilityKind>(Nullable);

    return None;
  }

  void setNullabilityAudited(NullabilityKind kind) {
    NullabilityAudited = true;
    Nullable = static_cast<unsigned>(kind);
  }

  const std::string &getType() const { return Type; }
  void setType(const std::string &type) { Type = type; }

  friend bool operator==(const VariableInfo &lhs, const VariableInfo &rhs) {
    return static_cast<const CommonEntityInfo &>(lhs) == rhs &&
           lhs.NullabilityAudited == rhs.NullabilityAudited &&
           lhs.Nullable == rhs.Nullable &&
           lhs.Type == rhs.Type;
  }

  friend bool operator!=(const VariableInfo &lhs, const VariableInfo &rhs) {
    return !(lhs == rhs);
  }

  friend VariableInfo &operator|=(VariableInfo &lhs,
                                  const VariableInfo &rhs) {
    static_cast<CommonEntityInfo &>(lhs) |= rhs;
    if (!lhs.NullabilityAudited && rhs.NullabilityAudited)
      lhs.setNullabilityAudited(*rhs.getNullability());
    if (lhs.Type.empty() && !rhs.Type.empty())
      lhs.Type = rhs.Type;
    return lhs;
  }
};

/// Describes API notes data for an Objective-C property.
class ObjCPropertyInfo : public VariableInfo {
  unsigned SwiftImportAsAccessorsSpecified : 1;
  unsigned SwiftImportAsAccessors : 1;

public:
  ObjCPropertyInfo()
      : VariableInfo(), SwiftImportAsAccessorsSpecified(false),
        SwiftImportAsAccessors(false) {}

  /// Merge class-wide information into the given property.
  friend ObjCPropertyInfo &operator|=(ObjCPropertyInfo &lhs,
                                      const ObjCContextInfo &rhs) {
    static_cast<VariableInfo &>(lhs) |= rhs;

    // Merge nullability.
    if (!lhs.getNullability()) {
      if (auto nullable = rhs.getDefaultNullability()) {
        lhs.setNullabilityAudited(*nullable);
      }
    }

    return lhs;
  }

  Optional<bool> getSwiftImportAsAccessors() const {
    if (SwiftImportAsAccessorsSpecified)
      return SwiftImportAsAccessors;
    return None;
  }
  void setSwiftImportAsAccessors(Optional<bool> value) {
    if (value.hasValue()) {
      SwiftImportAsAccessorsSpecified = true;
      SwiftImportAsAccessors = value.getValue();
    } else {
      SwiftImportAsAccessorsSpecified = false;
      SwiftImportAsAccessors = false;
    }
  }

  friend ObjCPropertyInfo &operator|=(ObjCPropertyInfo &lhs,
                                      const ObjCPropertyInfo &rhs) {
    lhs |= static_cast<const VariableInfo &>(rhs);
    if (!lhs.SwiftImportAsAccessorsSpecified &&
        rhs.SwiftImportAsAccessorsSpecified) {
      lhs.SwiftImportAsAccessorsSpecified = true;
      lhs.SwiftImportAsAccessors = rhs.SwiftImportAsAccessors;
    }
    return lhs;
  }

  friend bool operator==(const ObjCPropertyInfo &lhs,
                         const ObjCPropertyInfo &rhs) {
    return static_cast<const VariableInfo &>(lhs) == rhs &&
           lhs.getSwiftImportAsAccessors() == rhs.getSwiftImportAsAccessors();
  }
};

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
  ParamInfo() : VariableInfo(), NoEscapeSpecified(false), NoEscape(false),
      RawRetainCountConvention() { }

  Optional<bool> isNoEscape() const {
    if (!NoEscapeSpecified) return None;
    return NoEscape;
  }
  void setNoEscape(Optional<bool> noescape) {
    if (noescape) {
      NoEscapeSpecified = true;
      NoEscape = *noescape;
    } else {
      NoEscapeSpecified = false;
      NoEscape = false;
    }
  }

  Optional<RetainCountConventionKind> getRetainCountConvention() const {
    if (!RawRetainCountConvention)
      return None;
    return static_cast<RetainCountConventionKind>(RawRetainCountConvention - 1);
  }
  void setRetainCountConvention(Optional<RetainCountConventionKind> convention){
    if (convention)
      RawRetainCountConvention = static_cast<unsigned>(convention.getValue())+1;
    else
      RawRetainCountConvention = 0;
    assert(getRetainCountConvention() == convention && "bitfield too small");
  }

  friend ParamInfo &operator|=(ParamInfo &lhs, const ParamInfo &rhs) {
    static_cast<VariableInfo &>(lhs) |= rhs;
    if (!lhs.NoEscapeSpecified && rhs.NoEscapeSpecified) {
      lhs.NoEscapeSpecified = true;
      lhs.NoEscape = rhs.NoEscape;
    }
    if (!lhs.RawRetainCountConvention)
      lhs.RawRetainCountConvention = rhs.RawRetainCountConvention;
    return lhs;
  }

  friend bool operator==(const ParamInfo &lhs, const ParamInfo &rhs) {
    return static_cast<const VariableInfo &>(lhs) == rhs &&
           lhs.NoEscapeSpecified == rhs.NoEscapeSpecified &&
           lhs.NoEscape == rhs.NoEscape &&
           lhs.RawRetainCountConvention == rhs.RawRetainCountConvention;
  }

  friend bool operator!=(const ParamInfo &lhs, const ParamInfo &rhs) {
    return !(lhs == rhs);
  }
};

/// A temporary reference to an Objective-C selector, suitable for
/// referencing selector data on the stack.
///
/// Instances of this struct do not store references to any of the
/// data they contain; it is up to the user to ensure that the data
/// referenced by the identifier list persists.
struct ObjCSelectorRef {
  unsigned NumPieces;
  ArrayRef<StringRef> Identifiers;
};

/// API notes for a function or method.
class FunctionInfo : public CommonEntityInfo {
private:
  static unsigned const NullabilityKindMask = 0x3;
  static unsigned const NullabilityKindSize = 2;

public:
  /// Whether the signature has been audited with respect to nullability.
  /// If yes, we consider all types to be non-nullable unless otherwise noted.
  /// If this flag is not set, the pointer types are considered to have
  /// unknown nullability.
  unsigned NullabilityAudited : 1;

  /// Number of types whose nullability is encoded with the NullabilityPayload.
  unsigned NumAdjustedNullable : 8;

  /// A biased RetainCountConventionKind, where 0 means "unspecified".
  unsigned RawRetainCountConvention : 3;

  /// Stores the nullability of the return type and the parameters.
  //  NullabilityKindSize bits are used to encode the nullability. The info
  //  about the return type is stored at position 0, followed by the nullability
  //  of the parameters.
  uint64_t NullabilityPayload = 0;

  /// The result type of this function, as a C type.
  std::string ResultType;

  /// The function parameters.
  std::vector<ParamInfo> Params;

  FunctionInfo()
    : CommonEntityInfo(),
      NullabilityAudited(false),
      NumAdjustedNullable(0),
      RawRetainCountConvention() { }

  static unsigned getMaxNullabilityIndex() {
    return ((sizeof(NullabilityPayload) * CHAR_BIT)/NullabilityKindSize);
  }

  void addTypeInfo(unsigned index, NullabilityKind kind) {
    assert(index <= getMaxNullabilityIndex());
    assert(static_cast<unsigned>(kind) < NullabilityKindMask);
    NullabilityAudited = true;
    if (NumAdjustedNullable < index + 1)
      NumAdjustedNullable = index + 1;

    // Mask the bits.
    NullabilityPayload &= ~(NullabilityKindMask << (index * NullabilityKindSize));

    // Set the value.
    unsigned kindValue =
      (static_cast<unsigned>(kind)) << (index * NullabilityKindSize);
    NullabilityPayload |= kindValue;
  }

  /// Adds the return type info.
  void addReturnTypeInfo(NullabilityKind kind) {
    addTypeInfo(0, kind);
  }

  /// Adds the parameter type info.
  void addParamTypeInfo(unsigned index, NullabilityKind kind) {
    addTypeInfo(index + 1, kind);
  }

private:
  NullabilityKind getTypeInfo(unsigned index) const {
    assert(NullabilityAudited &&
           "Checking the type adjustment on non-audited method.");
    // If we don't have info about this parameter, return the default.
    if (index > NumAdjustedNullable)
      return NullabilityKind::NonNull;
    return static_cast<NullabilityKind>(( NullabilityPayload
                                          >> (index * NullabilityKindSize) )
                                         & NullabilityKindMask);
  }

public:
  NullabilityKind getParamTypeInfo(unsigned index) const {
    return getTypeInfo(index + 1);
  }
  
  NullabilityKind getReturnTypeInfo() const {
    return getTypeInfo(0);
  }

  Optional<RetainCountConventionKind> getRetainCountConvention() const {
    if (!RawRetainCountConvention)
      return None;
    return static_cast<RetainCountConventionKind>(RawRetainCountConvention - 1);
  }
  void setRetainCountConvention(Optional<RetainCountConventionKind> convention){
    if (convention)
      RawRetainCountConvention = static_cast<unsigned>(convention.getValue())+1;
    else
      RawRetainCountConvention = 0;
    assert(getRetainCountConvention() == convention && "bitfield too small");
  }

  friend bool operator==(const FunctionInfo &lhs, const FunctionInfo &rhs) {
    return static_cast<const CommonEntityInfo &>(lhs) == rhs &&
           lhs.NullabilityAudited == rhs.NullabilityAudited &&
           lhs.NumAdjustedNullable == rhs.NumAdjustedNullable &&
           lhs.NullabilityPayload == rhs.NullabilityPayload &&
           lhs.ResultType == rhs.ResultType &&
           lhs.Params == rhs.Params &&
           lhs.RawRetainCountConvention == rhs.RawRetainCountConvention;
  }

  friend bool operator!=(const FunctionInfo &lhs, const FunctionInfo &rhs) {
    return !(lhs == rhs);
  }

};

/// Describes API notes data for an Objective-C method.
class ObjCMethodInfo : public FunctionInfo {
public:
  /// Whether this is a designated initializer of its class.
  unsigned DesignatedInit : 1;

  /// Whether this is a required initializer.
  unsigned Required : 1;

  ObjCMethodInfo()
    : FunctionInfo(),
      DesignatedInit(false),
      Required(false) { }

  friend bool operator==(const ObjCMethodInfo &lhs, const ObjCMethodInfo &rhs) {
    return static_cast<const FunctionInfo &>(lhs) == rhs &&
           lhs.DesignatedInit == rhs.DesignatedInit &&
           lhs.Required == rhs.Required;
  }

  friend bool operator!=(const ObjCMethodInfo &lhs, const ObjCMethodInfo &rhs) {
    return !(lhs == rhs);
  }

  void mergePropInfoIntoSetter(const ObjCPropertyInfo &pInfo);

  void mergePropInfoIntoGetter(const ObjCPropertyInfo &pInfo);

  /// Merge class-wide information into the given method.
  friend ObjCMethodInfo &operator|=(ObjCMethodInfo &lhs,
                                    const ObjCContextInfo &rhs) {
    // Merge nullability.
    if (!lhs.NullabilityAudited) {
      if (auto nullable = rhs.getDefaultNullability()) {
        lhs.NullabilityAudited = true;
        lhs.addTypeInfo(0, *nullable);
      }
    }

    return lhs;
  }

  void dump(llvm::raw_ostream &os);
};

/// Describes API notes data for a global variable.
class GlobalVariableInfo : public VariableInfo {
public:
  GlobalVariableInfo() : VariableInfo() { }
};

/// Describes API notes data for a global function.
class GlobalFunctionInfo : public FunctionInfo {
public:
  GlobalFunctionInfo() : FunctionInfo() { }
};

/// Describes API notes data for an enumerator.
class EnumConstantInfo : public CommonEntityInfo {
public:
  EnumConstantInfo() : CommonEntityInfo() { }
};

/// The payload for an enum_extensibility attribute. This is a tri-state rather
/// than just a boolean because the presence of the attribute indicates
/// auditing.
enum class EnumExtensibilityKind {
  None,
  Open,
  Closed,
};

/// Describes API notes data for a tag.
class TagInfo : public CommonTypeInfo {
  unsigned HasFlagEnum : 1;
  unsigned IsFlagEnum : 1;
public:
  Optional<EnumExtensibilityKind> EnumExtensibility;

  Optional<bool> isFlagEnum() const {
    if (HasFlagEnum)
      return IsFlagEnum;
    return None;
  }
  void setFlagEnum(Optional<bool> Value) {
    if (Value.hasValue()) {
      HasFlagEnum = true;
      IsFlagEnum = Value.getValue();
    } else {
      HasFlagEnum = false;
    }
  }

  TagInfo() : CommonTypeInfo(), HasFlagEnum(0), IsFlagEnum(0) { }

  friend TagInfo &operator|=(TagInfo &lhs, const TagInfo &rhs) {
    lhs |= static_cast<const CommonTypeInfo &>(rhs);
    if (!lhs.HasFlagEnum && rhs.HasFlagEnum) {
      lhs.HasFlagEnum = true;
      lhs.IsFlagEnum = rhs.IsFlagEnum;
    }
    if (!lhs.EnumExtensibility.hasValue() && rhs.EnumExtensibility.hasValue())
      lhs.EnumExtensibility = rhs.EnumExtensibility;
    return lhs;
  }

  friend bool operator==(const TagInfo &lhs, const TagInfo &rhs) {
    return static_cast<const CommonTypeInfo &>(lhs) == rhs &&
           lhs.isFlagEnum() == rhs.isFlagEnum() &&
           lhs.EnumExtensibility == rhs.EnumExtensibility;
  }
};

/// The kind of a swift_wrapper/swift_newtype.
enum class SwiftWrapperKind {
  None,
  Struct,
  Enum
};

/// Describes API notes data for a typedef.
class TypedefInfo : public CommonTypeInfo {
public:
  Optional<SwiftWrapperKind> SwiftWrapper;

  TypedefInfo() : CommonTypeInfo() { }

  friend TypedefInfo &operator|=(TypedefInfo &lhs, const TypedefInfo &rhs) {
    lhs |= static_cast<const CommonTypeInfo &>(rhs);
    if (!lhs.SwiftWrapper.hasValue() && rhs.SwiftWrapper.hasValue())
      lhs.SwiftWrapper = rhs.SwiftWrapper;
    return lhs;
  }

  friend bool operator==(const TypedefInfo &lhs, const TypedefInfo &rhs) {
    return static_cast<const CommonTypeInfo &>(lhs) == rhs &&
           lhs.SwiftWrapper == rhs.SwiftWrapper;
  }
};

/// Descripts a series of options for a module
struct ModuleOptions {
  bool SwiftInferImportAsMember = false;
};

} // end namespace api_notes
} // end namespace clang

#endif // LLVM_CLANG_API_NOTES_TYPES_H
