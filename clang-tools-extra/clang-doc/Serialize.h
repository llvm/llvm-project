//===-- Serializer.h - ClangDoc Serializer ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the serializing functions fro the clang-doc tool. Given
// a particular declaration, it collects the appropriate information and returns
// a serialized bitcode string for the declaration.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SERIALIZE_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SERIALIZE_H

#include "Representation.h"
#include <string>

using namespace clang::comments;

namespace clang {
namespace doc {
namespace serialize {

// The first element will contain the relevant information about the declaration
// passed as parameter.
// The second element will contain the relevant information about the
// declaration's parent, it can be a NamespaceInfo or RecordInfo.
// Both elements can be nullptrs if the declaration shouldn't be handled.
// When the declaration is handled, the first element will be a nullptr for
// EnumDecl, FunctionDecl and CXXMethodDecl; they are only returned wrapped in
// its parent scope. For NamespaceDecl and RecordDecl both elements are not
// nullptr.
class Serializer {
public:
  Serializer() = default;

  std::pair<OwnedPtr<Info>, OwnedPtr<Info>> emitInfo(const NamespaceDecl *D,
                                                     const FullComment *FC,
                                                     Location Loc,
                                                     bool PublicOnly);

  std::pair<OwnedPtr<Info>, OwnedPtr<Info>> emitInfo(const RecordDecl *D,
                                                     const FullComment *FC,
                                                     Location Loc,
                                                     bool PublicOnly);

  std::pair<OwnedPtr<Info>, OwnedPtr<Info>> emitInfo(const EnumDecl *D,
                                                     const FullComment *FC,
                                                     Location Loc,
                                                     bool PublicOnly);

  std::pair<OwnedPtr<Info>, OwnedPtr<Info>> emitInfo(const FunctionDecl *D,
                                                     const FullComment *FC,
                                                     Location Loc,
                                                     bool PublicOnly);

  std::pair<OwnedPtr<Info>, OwnedPtr<Info>>
  emitInfo(const VarDecl *D, const FullComment *FC, int LineNumber,
           StringRef File, bool IsFileInRootDir, bool PublicOnly);

  std::pair<OwnedPtr<Info>, OwnedPtr<Info>> emitInfo(const CXXMethodDecl *D,
                                                     const FullComment *FC,
                                                     Location Loc,
                                                     bool PublicOnly);

  std::pair<OwnedPtr<Info>, OwnedPtr<Info>> emitInfo(const TypedefDecl *D,
                                                     const FullComment *FC,
                                                     Location Loc,
                                                     bool PublicOnly);

  std::pair<OwnedPtr<Info>, OwnedPtr<Info>> emitInfo(const TypeAliasDecl *D,
                                                     const FullComment *FC,
                                                     Location Loc,
                                                     bool PublicOnly);

  std::pair<OwnedPtr<Info>, OwnedPtr<Info>> emitInfo(const ConceptDecl *D,
                                                     const FullComment *FC,
                                                     const Location &Loc,
                                                     bool PublicOnly);

  std::pair<OwnedPtr<Info>, OwnedPtr<Info>> emitInfo(const VarDecl *D,
                                                     const FullComment *FC,
                                                     const Location &Loc,
                                                     bool PublicOnly);

private:
  void getTemplateParameters(const TemplateParameterList *TemplateParams,
                             llvm::raw_ostream &Stream);

  llvm::SmallString<256> getFunctionPrototype(const FunctionDecl *FuncDecl);

  llvm::SmallString<16> getTypeAlias(const TypeAliasDecl *Alias);

  llvm::SmallString<128>
  getInfoRelativePath(const llvm::SmallVectorImpl<doc::Reference> &Namespaces);

  llvm::SmallString<128> getInfoRelativePath(const Decl *D);

  std::string getSourceCode(const Decl *D, const SourceRange &R);

  void parseFullComment(const FullComment *C, CommentInfo &CI);

  SymbolID getUSRForDecl(const Decl *D);

  TagDecl *getTagDeclForType(const QualType &T);

  RecordDecl *getRecordDeclForType(const QualType &T);

  TypeInfo getTypeInfoForType(const QualType &T, const PrintingPolicy &Policy);

  bool isPublic(const clang::AccessSpecifier AS, const clang::Linkage Link);

  bool shouldSerializeInfo(bool PublicOnly, bool IsInAnonymousNamespace,
                           const NamedDecl *D);

  void InsertChild(ScopeChildren &Scope, const NamespaceInfo &Info);
  void InsertChild(ScopeChildren &Scope, const RecordInfo &Info);
  void InsertChild(ScopeChildren &Scope, EnumInfo Info);
  void InsertChild(ScopeChildren &Scope, FunctionInfo Info);
  void InsertChild(ScopeChildren &Scope, TypedefInfo Info);
  void InsertChild(ScopeChildren &Scope, ConceptInfo Info);
  void InsertChild(ScopeChildren &Scope, VarInfo Info);

  template <typename ChildType>
  OwnedPtr<Info> makeAndInsertIntoParent(ChildType Child);

  AccessSpecifier getFinalAccessSpecifier(AccessSpecifier FirstAS,
                                          AccessSpecifier SecondAS);

  void parseFields(RecordInfo &I, const RecordDecl *D, bool PublicOnly,
                   AccessSpecifier Access = AccessSpecifier::AS_public);

  void parseEnumerators(EnumInfo &I, const EnumDecl *D);

  void parseParameters(FunctionInfo &I, const FunctionDecl *D);

  void parseBases(RecordInfo &I, const CXXRecordDecl *D);

  void parseBases(RecordInfo &I, const CXXRecordDecl *D, bool IsFileInRootDir,
                  bool PublicOnly, bool IsParent,
                  AccessSpecifier ParentAccess = AccessSpecifier::AS_public);

  template <typename T>
  void populateParentNamespaces(llvm::SmallVector<Reference, 4> &Namespaces,
                                const T *D, bool &IsInAnonymousNamespace);

  void populateTemplateParameters(std::optional<TemplateInfo> &TemplateInfo,
                                  const clang::Decl *D);

  TemplateParamInfo convertTemplateArgToInfo(const clang::Decl *D,
                                             const TemplateArgument &Arg);

  bool isSupportedContext(Decl::Kind DeclKind);

  void findParent(Info &I, const Decl *D);

  template <typename T>
  void populateInfo(Info &I, const T *D, const FullComment *C,
                    bool &IsInAnonymousNamespace);

  template <typename T>
  void populateSymbolInfo(SymbolInfo &I, const T *D, const FullComment *C,
                          Location Loc, bool &IsInAnonymousNamespace);

  void handleCompoundConstraints(const Expr *Constraint,
                                 OwningVec<ConstraintInfo> &ConstraintInfos);

  void populateConstraints(TemplateInfo &I, const TemplateDecl *D);

  void populateFunctionInfo(FunctionInfo &I, const FunctionDecl *D,
                            const FullComment *FC, Location Loc,
                            bool &IsInAnonymousNamespace);

  template <typename T> void populateMemberTypeInfo(T &I, const Decl *D);

  void populateMemberTypeInfo(RecordInfo &I, AccessSpecifier &Access,
                              const DeclaratorDecl *D, bool IsStatic = false);

  void parseFriends(RecordInfo &RI, const CXXRecordDecl *D);

  void extractCommentFromDecl(const Decl *D, TypedefInfo &Info);
};

// Function to hash a given USR value for storage.
// As USRs (Unified Symbol Resolution) could be large, especially for functions
// with long type arguments, we use 160-bits SHA1(USR) values to
// guarantee the uniqueness of symbols while using a relatively small amount of
// memory (vs storing USRs directly).
SymbolID hashUSR(llvm::StringRef USR);

std::string serialize(OwnedPtr<Info> &I, DiagnosticsEngine &Diags);

} // namespace serialize
} // namespace doc
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_DOC_SERIALIZE_H
