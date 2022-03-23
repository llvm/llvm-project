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

  GlobalRecord(GVKind Kind, StringRef Name, StringRef USR, PresumedLoc Loc,
               const AvailabilityInfo &Availability, LinkageInfo Linkage,
               const DocComment &Comment, DeclarationFragments Declaration,
               DeclarationFragments SubHeading, FunctionSignature Signature)
      : APIRecord(RK_Global, Name, USR, Loc, Availability, Linkage, Comment,
                  Declaration, SubHeading),
        GlobalKind(Kind), Signature(Signature) {}

  static bool classof(const APIRecord *Record) {
    return Record->getKind() == RK_Global;
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

private:
  /// \brief A custom deleter used for ``std::unique_ptr`` to APIRecords stored
  /// in the BumpPtrAllocator.
  ///
  /// \tparam T the exact type of the APIRecord subclass.
  template <typename T> struct UniquePtrBumpPtrAllocatorDeleter {
    void operator()(T *Instance) { Instance->~T(); }
  };

public:
  /// A unique pointer to an APIRecord stored in the BumpPtrAllocator.
  ///
  /// \tparam T the exact type of the APIRecord subclass.
  template <typename T>
  using APIRecordUniquePtr =
      std::unique_ptr<T, UniquePtrBumpPtrAllocatorDeleter<T>>;

  /// A map to store the set of GlobalRecord%s with the declaration name as the
  /// key.
  using GlobalRecordMap =
      llvm::MapVector<StringRef, APIRecordUniquePtr<GlobalRecord>>;

  /// Get the target triple for the ExtractAPI invocation.
  const llvm::Triple &getTarget() const { return Target; }

  /// Get the language options used to parse the APIs.
  const LangOptions &getLangOpts() const { return LangOpts; }

  const GlobalRecordMap &getGlobals() const { return Globals; }

  /// Generate and store the USR of declaration \p D.
  ///
  /// Note: The USR string is stored in and owned by Allocator.
  ///
  /// \returns a StringRef of the generated USR string.
  StringRef recordUSR(const Decl *D);

  /// Copy \p String into the Allocator in this APISet.
  ///
  /// \returns a StringRef of the copied string in APISet::Allocator.
  StringRef copyString(StringRef String);

  APISet(const llvm::Triple &Target, const LangOptions &LangOpts)
      : Target(Target), LangOpts(LangOpts) {}

private:
  /// BumpPtrAllocator to store APIRecord%s and generated/copied strings.
  llvm::BumpPtrAllocator Allocator;

  const llvm::Triple Target;
  const LangOptions LangOpts;

  GlobalRecordMap Globals;
};

} // namespace extractapi
} // namespace clang

#endif // LLVM_CLANG_EXTRACTAPI_API_H
