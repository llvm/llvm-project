//===- ExtractAPI/API.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the APIRecord-based structs and the APISet class.
///
/// Clang ExtractAPI is a tool to collect API information from a given set of
/// header files. The structures in this file describe data representations of
/// the API information collected for various kinds of symbols.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_EXTRACTAPI_API_H
#define LLVM_CLANG_EXTRACTAPI_API_H

#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/RawCommentList.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/ExtractAPI/AvailabilityInfo.h"
#include "clang/ExtractAPI/DeclarationFragments.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <type_traits>

namespace clang {
namespace extractapi {

/// DocComment is a vector of RawComment::CommentLine.
///
/// Each line represents one line of striped documentation comment,
/// with source range information. This simplifies calculating the source
/// location of a character in the doc comment for pointing back to the source
/// file.
/// e.g.
/// \code
///   /// This is a documentation comment
///       ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'  First line.
///   ///     with multiple lines.
///       ^~~~~~~~~~~~~~~~~~~~~~~'         Second line.
/// \endcode
using DocComment = std::vector<RawComment::CommentLine>;

// Classes deriving from APIRecord need to have Name be the first constructor
// argument. This is so that they are compatible with `addTopLevelRecord`
// defined in API.cpp
/// The base representation of an API record. Holds common symbol information.
struct APIRecord {
  StringRef Name;
  StringRef USR;
  PresumedLoc Location;
  AvailabilityInfo Availability;
  LinkageInfo Linkage;

  /// Documentation comment lines attached to this symbol declaration.
  DocComment Comment;

  /// Declaration fragments of this symbol declaration.
  DeclarationFragments Declaration;

  /// SubHeading provides a more detailed representation than the plain
  /// declaration name.
  ///
  /// SubHeading is an array of declaration fragments of tagged declaration
  /// name, with potentially more tokens (for example the \c +/- symbol for
  /// Objective-C class/instance methods).
  DeclarationFragments SubHeading;

  /// Discriminator for LLVM-style RTTI (dyn_cast<> et al.)
  enum RecordKind {
    RK_Global,
    RK_EnumConstant,
    RK_Enum,
    RK_StructField,
    RK_Struct,
    RK_ObjCProperty,
    RK_ObjCIvar,
    RK_ObjCMethod,
    RK_ObjCInterface,
    RK_ObjCCategory,
    RK_ObjCProtocol,
    RK_MacroDefinition,
    RK_Typedef,
  };

private:
  const RecordKind Kind;

public:
  RecordKind getKind() const { return Kind; }

  APIRecord() = delete;

  APIRecord(RecordKind Kind, StringRef Name, StringRef USR,
            PresumedLoc Location, const AvailabilityInfo &Availability,
            LinkageInfo Linkage, const DocComment &Comment,
            DeclarationFragments Declaration, DeclarationFragments SubHeading)
      : Name(Name), USR(USR), Location(Location), Availability(Availability),
        Linkage(Linkage), Comment(Comment), Declaration(Declaration),
        SubHeading(SubHeading), Kind(Kind) {}

  // Pure virtual destructor to make APIRecord abstract
  virtual ~APIRecord() = 0;
};

/// The kind of a global record.
enum class GVKind : uint8_t {
  Unknown = 0,
  Variable = 1,
  Function = 2,
};

/// This holds information associated with global variables or functions.
struct GlobalRecord : APIRecord {
  GVKind GlobalKind;

  /// The function signature of the record if it is a function.
  FunctionSignature Signature;

  GlobalRecord(StringRef Name, StringRef USR, PresumedLoc Loc,
               const AvailabilityInfo &Availability, LinkageInfo Linkage,
               const DocComment &Comment, DeclarationFragments Declaration,
               DeclarationFragments SubHeading, GVKind Kind,
               FunctionSignature Signature)
      : APIRecord(RK_Global, Name, USR, Loc, Availability, Linkage, Comment,
                  Declaration, SubHeading),
        GlobalKind(Kind), Signature(Signature) {}

  static bool classof(const APIRecord *Record) {
    return Record->getKind() == RK_Global;
  }

private:
  virtual void anchor();
};

/// This holds information associated with enum constants.
struct EnumConstantRecord : APIRecord {
  EnumConstantRecord(StringRef Name, StringRef USR, PresumedLoc Loc,
                     const AvailabilityInfo &Availability,
                     const DocComment &Comment,
                     DeclarationFragments Declaration,
                     DeclarationFragments SubHeading)
      : APIRecord(RK_EnumConstant, Name, USR, Loc, Availability,
                  LinkageInfo::none(), Comment, Declaration, SubHeading) {}

  static bool classof(const APIRecord *Record) {
    return Record->getKind() == RK_EnumConstant;
  }

private:
  virtual void anchor();
};

/// This holds information associated with enums.
struct EnumRecord : APIRecord {
  SmallVector<std::unique_ptr<EnumConstantRecord>> Constants;

  EnumRecord(StringRef Name, StringRef USR, PresumedLoc Loc,
             const AvailabilityInfo &Availability, const DocComment &Comment,
             DeclarationFragments Declaration, DeclarationFragments SubHeading)
      : APIRecord(RK_Enum, Name, USR, Loc, Availability, LinkageInfo::none(),
                  Comment, Declaration, SubHeading) {}

  static bool classof(const APIRecord *Record) {
    return Record->getKind() == RK_Enum;
  }

private:
  virtual void anchor();
};

/// This holds information associated with struct fields.
struct StructFieldRecord : APIRecord {
  StructFieldRecord(StringRef Name, StringRef USR, PresumedLoc Loc,
                    const AvailabilityInfo &Availability,
                    const DocComment &Comment, DeclarationFragments Declaration,
                    DeclarationFragments SubHeading)
      : APIRecord(RK_StructField, Name, USR, Loc, Availability,
                  LinkageInfo::none(), Comment, Declaration, SubHeading) {}

  static bool classof(const APIRecord *Record) {
    return Record->getKind() == RK_StructField;
  }

private:
  virtual void anchor();
};

/// This holds information associated with structs.
struct StructRecord : APIRecord {
  SmallVector<std::unique_ptr<StructFieldRecord>> Fields;

  StructRecord(StringRef Name, StringRef USR, PresumedLoc Loc,
               const AvailabilityInfo &Availability, const DocComment &Comment,
               DeclarationFragments Declaration,
               DeclarationFragments SubHeading)
      : APIRecord(RK_Struct, Name, USR, Loc, Availability, LinkageInfo::none(),
                  Comment, Declaration, SubHeading) {}

  static bool classof(const APIRecord *Record) {
    return Record->getKind() == RK_Struct;
  }

private:
  virtual void anchor();
};

/// This holds information associated with Objective-C properties.
struct ObjCPropertyRecord : APIRecord {
  /// The attributes associated with an Objective-C property.
  enum AttributeKind : unsigned {
    NoAttr = 0,
    ReadOnly = 1,
    Class = 1 << 1,
    Dynamic = 1 << 2,
  };

  AttributeKind Attributes;
  StringRef GetterName;
  StringRef SetterName;
  bool IsOptional;

  ObjCPropertyRecord(StringRef Name, StringRef USR, PresumedLoc Loc,
                     const AvailabilityInfo &Availability,
                     const DocComment &Comment,
                     DeclarationFragments Declaration,
                     DeclarationFragments SubHeading, AttributeKind Attributes,
                     StringRef GetterName, StringRef SetterName,
                     bool IsOptional)
      : APIRecord(RK_ObjCProperty, Name, USR, Loc, Availability,
                  LinkageInfo::none(), Comment, Declaration, SubHeading),
        Attributes(Attributes), GetterName(GetterName), SetterName(SetterName),
        IsOptional(IsOptional) {}

  bool isReadOnly() const { return Attributes & ReadOnly; }
  bool isDynamic() const { return Attributes & Dynamic; }
  bool isClassProperty() const { return Attributes & Class; }

  static bool classof(const APIRecord *Record) {
    return Record->getKind() == RK_ObjCProperty;
  }

private:
  virtual void anchor();
};

/// This holds information associated with Objective-C instance variables.
struct ObjCInstanceVariableRecord : APIRecord {
  using AccessControl = ObjCIvarDecl::AccessControl;
  AccessControl Access;

  ObjCInstanceVariableRecord(StringRef Name, StringRef USR, PresumedLoc Loc,
                             const AvailabilityInfo &Availability,
                             const DocComment &Comment,
                             DeclarationFragments Declaration,
                             DeclarationFragments SubHeading,
                             AccessControl Access)
      : APIRecord(RK_ObjCIvar, Name, USR, Loc, Availability,
                  LinkageInfo::none(), Comment, Declaration, SubHeading),
        Access(Access) {}

  static bool classof(const APIRecord *Record) {
    return Record->getKind() == RK_ObjCIvar;
  }

private:
  virtual void anchor();
};

/// This holds information associated with Objective-C methods.
struct ObjCMethodRecord : APIRecord {
  FunctionSignature Signature;
  bool IsInstanceMethod;

  ObjCMethodRecord(StringRef Name, StringRef USR, PresumedLoc Loc,
                   const AvailabilityInfo &Availability,
                   const DocComment &Comment, DeclarationFragments Declaration,
                   DeclarationFragments SubHeading, FunctionSignature Signature,
                   bool IsInstanceMethod)
      : APIRecord(RK_ObjCMethod, Name, USR, Loc, Availability,
                  LinkageInfo::none(), Comment, Declaration, SubHeading),
        Signature(Signature), IsInstanceMethod(IsInstanceMethod) {}

  static bool classof(const APIRecord *Record) {
    return Record->getKind() == RK_ObjCMethod;
  }

private:
  virtual void anchor();
};

/// This represents a reference to another symbol that might come from external
/// sources.
struct SymbolReference {
  StringRef Name;
  StringRef USR;

  /// The source project/module/product of the referred symbol.
  StringRef Source;

  SymbolReference() = default;
  SymbolReference(StringRef Name, StringRef USR = "", StringRef Source = "")
      : Name(Name), USR(USR), Source(Source) {}
  SymbolReference(const APIRecord &Record)
      : Name(Record.Name), USR(Record.USR) {}

  /// Determine if this SymbolReference is empty.
  ///
  /// \returns true if and only if all \c Name, \c USR, and \c Source is empty.
  bool empty() const { return Name.empty() && USR.empty() && Source.empty(); }
};

/// The base representation of an Objective-C container record. Holds common
/// information associated with Objective-C containers.
struct ObjCContainerRecord : APIRecord {
  SmallVector<std::unique_ptr<ObjCMethodRecord>> Methods;
  SmallVector<std::unique_ptr<ObjCPropertyRecord>> Properties;
  SmallVector<std::unique_ptr<ObjCInstanceVariableRecord>> Ivars;
  SmallVector<SymbolReference> Protocols;

  ObjCContainerRecord() = delete;

  ObjCContainerRecord(RecordKind Kind, StringRef Name, StringRef USR,
                      PresumedLoc Loc, const AvailabilityInfo &Availability,
                      LinkageInfo Linkage, const DocComment &Comment,
                      DeclarationFragments Declaration,
                      DeclarationFragments SubHeading)
      : APIRecord(Kind, Name, USR, Loc, Availability, Linkage, Comment,
                  Declaration, SubHeading) {}

  virtual ~ObjCContainerRecord() = 0;
};

/// This holds information associated with Objective-C categories.
struct ObjCCategoryRecord : ObjCContainerRecord {
  SymbolReference Interface;

  ObjCCategoryRecord(StringRef Name, StringRef USR, PresumedLoc Loc,
                     const AvailabilityInfo &Availability,
                     const DocComment &Comment,
                     DeclarationFragments Declaration,
                     DeclarationFragments SubHeading, SymbolReference Interface)
      : ObjCContainerRecord(RK_ObjCCategory, Name, USR, Loc, Availability,
                            LinkageInfo::none(), Comment, Declaration,
                            SubHeading),
        Interface(Interface) {}

  static bool classof(const APIRecord *Record) {
    return Record->getKind() == RK_ObjCCategory;
  }

private:
  virtual void anchor();
};

/// This holds information associated with Objective-C interfaces/classes.
struct ObjCInterfaceRecord : ObjCContainerRecord {
  SymbolReference SuperClass;
  // ObjCCategoryRecord%s are stored in and owned by APISet.
  SmallVector<ObjCCategoryRecord *> Categories;

  ObjCInterfaceRecord(StringRef Name, StringRef USR, PresumedLoc Loc,
                      const AvailabilityInfo &Availability, LinkageInfo Linkage,
                      const DocComment &Comment,
                      DeclarationFragments Declaration,
                      DeclarationFragments SubHeading,
                      SymbolReference SuperClass)
      : ObjCContainerRecord(RK_ObjCInterface, Name, USR, Loc, Availability,
                            Linkage, Comment, Declaration, SubHeading),
        SuperClass(SuperClass) {}

  static bool classof(const APIRecord *Record) {
    return Record->getKind() == RK_ObjCInterface;
  }

private:
  virtual void anchor();
};

/// This holds information associated with Objective-C protocols.
struct ObjCProtocolRecord : ObjCContainerRecord {
  ObjCProtocolRecord(StringRef Name, StringRef USR, PresumedLoc Loc,
                     const AvailabilityInfo &Availability,
                     const DocComment &Comment,
                     DeclarationFragments Declaration,
                     DeclarationFragments SubHeading)
      : ObjCContainerRecord(RK_ObjCProtocol, Name, USR, Loc, Availability,
                            LinkageInfo::none(), Comment, Declaration,
                            SubHeading) {}

  static bool classof(const APIRecord *Record) {
    return Record->getKind() == RK_ObjCProtocol;
  }

private:
  virtual void anchor();
};

/// This holds information associated with macro definitions.
struct MacroDefinitionRecord : APIRecord {
  MacroDefinitionRecord(StringRef Name, StringRef USR, PresumedLoc Loc,
                        DeclarationFragments Declaration,
                        DeclarationFragments SubHeading)
      : APIRecord(RK_MacroDefinition, Name, USR, Loc, AvailabilityInfo(),
                  LinkageInfo(), {}, Declaration, SubHeading) {}

  static bool classof(const APIRecord *Record) {
    return Record->getKind() == RK_MacroDefinition;
  }

private:
  virtual void anchor();
};

/// This holds information associated with typedefs.
///
/// Note: Typedefs for anonymous enums and structs typically don't get emitted
/// by the serializers but still get a TypedefRecord. Instead we use the
/// typedef name as a name for the underlying anonymous struct or enum.
struct TypedefRecord : APIRecord {
  SymbolReference UnderlyingType;

  TypedefRecord(StringRef Name, StringRef USR, PresumedLoc Loc,
                const AvailabilityInfo &Availability, const DocComment &Comment,
                DeclarationFragments Declaration,
                DeclarationFragments SubHeading, SymbolReference UnderlyingType)
      : APIRecord(RK_Typedef, Name, USR, Loc, Availability, LinkageInfo(),
                  Comment, Declaration, SubHeading),
        UnderlyingType(UnderlyingType) {}

  static bool classof(const APIRecord *Record) {
    return Record->getKind() == RK_Typedef;
  }

private:
  virtual void anchor();
};

/// APISet holds the set of API records collected from given inputs.
class APISet {
public:
  /// Create and add a GlobalRecord of kind \p Kind into the API set.
  ///
  /// Note: the caller is responsible for keeping the StringRef \p Name and
  /// \p USR alive. APISet::copyString provides a way to copy strings into
  /// APISet itself, and APISet::recordUSR(const Decl *D) is a helper method
  /// to generate the USR for \c D and keep it alive in APISet.
  GlobalRecord *addGlobal(GVKind Kind, StringRef Name, StringRef USR,
                          PresumedLoc Loc, const AvailabilityInfo &Availability,
                          LinkageInfo Linkage, const DocComment &Comment,
                          DeclarationFragments Declaration,
                          DeclarationFragments SubHeading,
                          FunctionSignature Signature);

  /// Create and add a global variable record into the API set.
  ///
  /// Note: the caller is responsible for keeping the StringRef \p Name and
  /// \p USR alive. APISet::copyString provides a way to copy strings into
  /// APISet itself, and APISet::recordUSR(const Decl *D) is a helper method
  /// to generate the USR for \c D and keep it alive in APISet.
  GlobalRecord *addGlobalVar(StringRef Name, StringRef USR, PresumedLoc Loc,
                             const AvailabilityInfo &Availability,
                             LinkageInfo Linkage, const DocComment &Comment,
                             DeclarationFragments Declaration,
                             DeclarationFragments SubHeading);

  /// Create and add a function record into the API set.
  ///
  /// Note: the caller is responsible for keeping the StringRef \p Name and
  /// \p USR alive. APISet::copyString provides a way to copy strings into
  /// APISet itself, and APISet::recordUSR(const Decl *D) is a helper method
  /// to generate the USR for \c D and keep it alive in APISet.
  GlobalRecord *addFunction(StringRef Name, StringRef USR, PresumedLoc Loc,
                            const AvailabilityInfo &Availability,
                            LinkageInfo Linkage, const DocComment &Comment,
                            DeclarationFragments Declaration,
                            DeclarationFragments SubHeading,
                            FunctionSignature Signature);

  /// Create and add an enum constant record into the API set.
  ///
  /// Note: the caller is responsible for keeping the StringRef \p Name and
  /// \p USR alive. APISet::copyString provides a way to copy strings into
  /// APISet itself, and APISet::recordUSR(const Decl *D) is a helper method
  /// to generate the USR for \c D and keep it alive in APISet.
  EnumConstantRecord *addEnumConstant(EnumRecord *Enum, StringRef Name,
                                      StringRef USR, PresumedLoc Loc,
                                      const AvailabilityInfo &Availability,
                                      const DocComment &Comment,
                                      DeclarationFragments Declaration,
                                      DeclarationFragments SubHeading);

  /// Create and add an enum record into the API set.
  ///
  /// Note: the caller is responsible for keeping the StringRef \p Name and
  /// \p USR alive. APISet::copyString provides a way to copy strings into
  /// APISet itself, and APISet::recordUSR(const Decl *D) is a helper method
  /// to generate the USR for \c D and keep it alive in APISet.
  EnumRecord *addEnum(StringRef Name, StringRef USR, PresumedLoc Loc,
                      const AvailabilityInfo &Availability,
                      const DocComment &Comment,
                      DeclarationFragments Declaration,
                      DeclarationFragments SubHeading);

  /// Create and add a struct field record into the API set.
  ///
  /// Note: the caller is responsible for keeping the StringRef \p Name and
  /// \p USR alive. APISet::copyString provides a way to copy strings into
  /// APISet itself, and APISet::recordUSR(const Decl *D) is a helper method
  /// to generate the USR for \c D and keep it alive in APISet.
  StructFieldRecord *addStructField(StructRecord *Struct, StringRef Name,
                                    StringRef USR, PresumedLoc Loc,
                                    const AvailabilityInfo &Availability,
                                    const DocComment &Comment,
                                    DeclarationFragments Declaration,
                                    DeclarationFragments SubHeading);

  /// Create and add a struct record into the API set.
  ///
  /// Note: the caller is responsible for keeping the StringRef \p Name and
  /// \p USR alive. APISet::copyString provides a way to copy strings into
  /// APISet itself, and APISet::recordUSR(const Decl *D) is a helper method
  /// to generate the USR for \c D and keep it alive in APISet.
  StructRecord *addStruct(StringRef Name, StringRef USR, PresumedLoc Loc,
                          const AvailabilityInfo &Availability,
                          const DocComment &Comment,
                          DeclarationFragments Declaration,
                          DeclarationFragments SubHeading);

  /// Create and add an Objective-C category record into the API set.
  ///
  /// Note: the caller is responsible for keeping the StringRef \p Name and
  /// \p USR alive. APISet::copyString provides a way to copy strings into
  /// APISet itself, and APISet::recordUSR(const Decl *D) is a helper method
  /// to generate the USR for \c D and keep it alive in APISet.
  ObjCCategoryRecord *
  addObjCCategory(StringRef Name, StringRef USR, PresumedLoc Loc,
                  const AvailabilityInfo &Availability,
                  const DocComment &Comment, DeclarationFragments Declaration,
                  DeclarationFragments SubHeading, SymbolReference Interface);

  /// Create and add an Objective-C interface record into the API set.
  ///
  /// Note: the caller is responsible for keeping the StringRef \p Name and
  /// \p USR alive. APISet::copyString provides a way to copy strings into
  /// APISet itself, and APISet::recordUSR(const Decl *D) is a helper method
  /// to generate the USR for \c D and keep it alive in APISet.
  ObjCInterfaceRecord *
  addObjCInterface(StringRef Name, StringRef USR, PresumedLoc Loc,
                   const AvailabilityInfo &Availability, LinkageInfo Linkage,
                   const DocComment &Comment, DeclarationFragments Declaration,
                   DeclarationFragments SubHeading, SymbolReference SuperClass);

  /// Create and add an Objective-C method record into the API set.
  ///
  /// Note: the caller is responsible for keeping the StringRef \p Name and
  /// \p USR alive. APISet::copyString provides a way to copy strings into
  /// APISet itself, and APISet::recordUSR(const Decl *D) is a helper method
  /// to generate the USR for \c D and keep it alive in APISet.
  ObjCMethodRecord *
  addObjCMethod(ObjCContainerRecord *Container, StringRef Name, StringRef USR,
                PresumedLoc Loc, const AvailabilityInfo &Availability,
                const DocComment &Comment, DeclarationFragments Declaration,
                DeclarationFragments SubHeading, FunctionSignature Signature,
                bool IsInstanceMethod);

  /// Create and add an Objective-C property record into the API set.
  ///
  /// Note: the caller is responsible for keeping the StringRef \p Name and
  /// \p USR alive. APISet::copyString provides a way to copy strings into
  /// APISet itself, and APISet::recordUSR(const Decl *D) is a helper method
  /// to generate the USR for \c D and keep it alive in APISet.
  ObjCPropertyRecord *
  addObjCProperty(ObjCContainerRecord *Container, StringRef Name, StringRef USR,
                  PresumedLoc Loc, const AvailabilityInfo &Availability,
                  const DocComment &Comment, DeclarationFragments Declaration,
                  DeclarationFragments SubHeading,
                  ObjCPropertyRecord::AttributeKind Attributes,
                  StringRef GetterName, StringRef SetterName, bool IsOptional);

  /// Create and add an Objective-C instance variable record into the API set.
  ///
  /// Note: the caller is responsible for keeping the StringRef \p Name and
  /// \p USR alive. APISet::copyString provides a way to copy strings into
  /// APISet itself, and APISet::recordUSR(const Decl *D) is a helper method
  /// to generate the USR for \c D and keep it alive in APISet.
  ObjCInstanceVariableRecord *addObjCInstanceVariable(
      ObjCContainerRecord *Container, StringRef Name, StringRef USR,
      PresumedLoc Loc, const AvailabilityInfo &Availability,
      const DocComment &Comment, DeclarationFragments Declaration,
      DeclarationFragments SubHeading,
      ObjCInstanceVariableRecord::AccessControl Access);

  /// Create and add an Objective-C protocol record into the API set.
  ///
  /// Note: the caller is responsible for keeping the StringRef \p Name and
  /// \p USR alive. APISet::copyString provides a way to copy strings into
  /// APISet itself, and APISet::recordUSR(const Decl *D) is a helper method
  /// to generate the USR for \c D and keep it alive in APISet.
  ObjCProtocolRecord *addObjCProtocol(StringRef Name, StringRef USR,
                                      PresumedLoc Loc,
                                      const AvailabilityInfo &Availability,
                                      const DocComment &Comment,
                                      DeclarationFragments Declaration,
                                      DeclarationFragments SubHeading);

  /// Create a macro definition record into the API set.
  ///
  /// Note: the caller is responsible for keeping the StringRef \p Name and
  /// \p USR alive. APISet::copyString provides a way to copy strings into
  /// APISet itself, and APISet::recordUSRForMacro(StringRef Name,
  /// SourceLocation SL, const SourceManager &SM) is a helper method to generate
  /// the USR for the macro and keep it alive in APISet.
  MacroDefinitionRecord *addMacroDefinition(StringRef Name, StringRef USR,
                                            PresumedLoc Loc,
                                            DeclarationFragments Declaration,
                                            DeclarationFragments SubHeading);

  /// Create a typedef record into the API set.
  ///
  /// Note: the caller is responsible for keeping the StringRef \p Name and
  /// \p USR alive. APISet::copyString provides a way to copy strings into
  /// APISet itself, and APISet::recordUSR(const Decl *D) is a helper method
  /// to generate the USR for \c D and keep it alive in APISet.
  TypedefRecord *addTypedef(StringRef Name, StringRef USR, PresumedLoc Loc,
                            const AvailabilityInfo &Availability,
                            const DocComment &Comment,
                            DeclarationFragments Declaration,
                            DeclarationFragments SubHeading,
                            SymbolReference UnderlyingType);

  /// A mapping type to store a set of APIRecord%s with the declaration name as
  /// the key.
  template <typename RecordTy,
            typename =
                std::enable_if_t<std::is_base_of<APIRecord, RecordTy>::value>>
  using RecordMap = llvm::MapVector<StringRef, std::unique_ptr<RecordTy>>;

  /// Get the target triple for the ExtractAPI invocation.
  const llvm::Triple &getTarget() const { return Target; }

  /// Get the language used by the APIs.
  Language getLanguage() const { return Lang; }

  const RecordMap<GlobalRecord> &getGlobals() const { return Globals; }
  const RecordMap<EnumRecord> &getEnums() const { return Enums; }
  const RecordMap<StructRecord> &getStructs() const { return Structs; }
  const RecordMap<ObjCCategoryRecord> &getObjCCategories() const {
    return ObjCCategories;
  }
  const RecordMap<ObjCInterfaceRecord> &getObjCInterfaces() const {
    return ObjCInterfaces;
  }
  const RecordMap<ObjCProtocolRecord> &getObjCProtocols() const {
    return ObjCProtocols;
  }
  const RecordMap<MacroDefinitionRecord> &getMacros() const { return Macros; }
  const RecordMap<TypedefRecord> &getTypedefs() const { return Typedefs; }

  /// Generate and store the USR of declaration \p D.
  ///
  /// Note: The USR string is stored in and owned by Allocator.
  ///
  /// \returns a StringRef of the generated USR string.
  StringRef recordUSR(const Decl *D);

  /// Generate and store the USR for a macro \p Name.
  ///
  /// Note: The USR string is stored in and owned by Allocator.
  ///
  /// \returns a StringRef to the generate USR string.
  StringRef recordUSRForMacro(StringRef Name, SourceLocation SL,
                              const SourceManager &SM);

  /// Copy \p String into the Allocator in this APISet.
  ///
  /// \returns a StringRef of the copied string in APISet::Allocator.
  StringRef copyString(StringRef String);

  APISet(const llvm::Triple &Target, Language Lang)
      : Target(Target), Lang(Lang) {}

private:
  /// BumpPtrAllocator to store generated/copied strings.
  ///
  /// Note: The main use for this is being able to deduplicate strings.
  llvm::BumpPtrAllocator StringAllocator;

  const llvm::Triple Target;
  const Language Lang;

  RecordMap<GlobalRecord> Globals;
  RecordMap<EnumRecord> Enums;
  RecordMap<StructRecord> Structs;
  RecordMap<ObjCCategoryRecord> ObjCCategories;
  RecordMap<ObjCInterfaceRecord> ObjCInterfaces;
  RecordMap<ObjCProtocolRecord> ObjCProtocols;
  RecordMap<MacroDefinitionRecord> Macros;
  RecordMap<TypedefRecord> Typedefs;
};

} // namespace extractapi
} // namespace clang

#endif // LLVM_CLANG_EXTRACTAPI_API_H
