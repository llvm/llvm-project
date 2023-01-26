//===- ExtractAPI/ExtractAPIVisitor.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the ExtractAPVisitor AST visitation interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_EXTRACTAPI_EXTRACT_API_VISITOR_H
#define LLVM_CLANG_EXTRACTAPI_EXTRACT_API_VISITOR_H

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/ExtractAPI/API.h"
#include "llvm/ADT/FunctionExtras.h"

namespace clang {
namespace extractapi {

/// The RecursiveASTVisitor to traverse symbol declarations and collect API
/// information.
class ExtractAPIVisitor : public RecursiveASTVisitor<ExtractAPIVisitor> {
public:
  ExtractAPIVisitor(ASTContext &Context,
                    llvm::unique_function<bool(SourceLocation)> LocationChecker,
                    APISet &API)
      : Context(Context), API(API),
        LocationChecker(std::move(LocationChecker)) {}

  const APISet &getAPI() const { return API; }

  bool VisitVarDecl(const VarDecl *Decl);

  bool VisitFunctionDecl(const FunctionDecl *Decl);

  bool VisitEnumDecl(const EnumDecl *Decl);

  bool VisitRecordDecl(const RecordDecl *Decl);

  bool VisitObjCInterfaceDecl(const ObjCInterfaceDecl *Decl);

  bool VisitObjCProtocolDecl(const ObjCProtocolDecl *Decl);

  bool VisitTypedefNameDecl(const TypedefNameDecl *Decl);

  bool VisitObjCCategoryDecl(const ObjCCategoryDecl *Decl);

private:
  /// Collect API information for the enum constants and associate with the
  /// parent enum.
  void recordEnumConstants(EnumRecord *EnumRecord,
                           const EnumDecl::enumerator_range Constants);

  /// Collect API information for the struct fields and associate with the
  /// parent struct.
  void recordStructFields(StructRecord *StructRecord,
                          const RecordDecl::field_range Fields);

  /// Collect API information for the Objective-C methods and associate with the
  /// parent container.
  void recordObjCMethods(ObjCContainerRecord *Container,
                         const ObjCContainerDecl::method_range Methods);

  void recordObjCProperties(ObjCContainerRecord *Container,
                            const ObjCContainerDecl::prop_range Properties);

  void recordObjCInstanceVariables(
      ObjCContainerRecord *Container,
      const llvm::iterator_range<
          DeclContext::specific_decl_iterator<ObjCIvarDecl>>
          Ivars);

  void recordObjCProtocols(ObjCContainerRecord *Container,
                           ObjCInterfaceDecl::protocol_range Protocols);
  ASTContext &Context;
  APISet &API;
  llvm::unique_function<bool(SourceLocation)> LocationChecker;
};

} // namespace extractapi
} // namespace clang

#endif // LLVM_CLANG_EXTRACTAPI_EXTRACT_API_VISITOR_H
