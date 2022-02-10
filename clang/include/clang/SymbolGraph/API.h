//===- SymbolGraph/API.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines SymbolGraph API records.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SYMBOLGRAPH_API_H
#define LLVM_CLANG_SYMBOLGRAPH_API_H

#include "clang/AST/Decl.h"
#include "clang/AST/RawCommentList.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/SymbolGraph/AvailabilityInfo.h"
#include "clang/SymbolGraph/DeclarationFragments.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Casting.h"

namespace clang {
namespace symbolgraph {

using DocComment = std::vector<RawComment::CommentLine>;

struct APIRecord {
  StringRef Name;
  StringRef USR;
  PresumedLoc Location;
  AvailabilityInfo Availability;
  LinkageInfo Linkage;
  DocComment Comment;
  DeclarationFragments Declaration;
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

enum class GVKind : uint8_t {
  Unknown = 0,
  Variable = 1,
  Function = 2,
};

struct GlobalRecord : APIRecord {
  GVKind GlobalKind;
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
};

class API {
public:
  API(const llvm::Triple &Target, const LangOptions &LangOpts)
      : Target(Target), LangOpts(LangOpts) {}

  const llvm::Triple &getTarget() const { return Target; }
  const LangOptions &getLangOpts() const { return LangOpts; }

  GlobalRecord *addGlobal(GVKind Kind, StringRef Name, StringRef USR,
                          PresumedLoc Loc, const AvailabilityInfo &Availability,
                          LinkageInfo Linkage, const DocComment &Comment,
                          DeclarationFragments Declaration,
                          DeclarationFragments SubHeading,
                          FunctionSignature Signature);

  GlobalRecord *addGlobalVar(StringRef Name, StringRef USR, PresumedLoc Loc,
                             const AvailabilityInfo &Availability,
                             LinkageInfo Linkage, const DocComment &Comment,
                             DeclarationFragments Declaration,
                             DeclarationFragments SubHeading);

  GlobalRecord *addFunction(StringRef Name, StringRef USR, PresumedLoc Loc,
                            const AvailabilityInfo &Availability,
                            LinkageInfo Linkage, const DocComment &Comment,
                            DeclarationFragments Declaration,
                            DeclarationFragments SubHeading,
                            FunctionSignature Signature);

  StringRef recordUSR(const Decl *D);
  StringRef copyString(StringRef String, llvm::BumpPtrAllocator &Allocator);
  StringRef copyString(StringRef String);

  using GlobalRecordMap = llvm::MapVector<StringRef, GlobalRecord *>;

  const GlobalRecordMap &getGlobals() const { return Globals; }

private:
  llvm::BumpPtrAllocator Allocator;
  const llvm::Triple Target;
  const LangOptions LangOpts;

  GlobalRecordMap Globals;
};

} // namespace symbolgraph
} // namespace clang

#endif // LLVM_CLANG_SYMBOLGRAPH_API_H
