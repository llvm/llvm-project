//===--- MakeUniqueCheck.cpp - clang-tidy----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MakeUniqueCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

MakeUniqueCheck::MakeUniqueCheck(StringRef Name,
                                 clang::tidy::ClangTidyContext *Context)
    : MakeSmartPtrCheck(Name, Context, "std::make_unique") {}

MakeUniqueCheck::SmartPtrTypeMatcher
MakeUniqueCheck::getSmartPointerTypeMatcher() const {
  return qualType(hasDeclaration(classTemplateSpecializationDecl(
      hasName("::std::unique_ptr"), templateArgumentCountIs(2),
      hasTemplateArgument(
          0, templateArgument(refersToType(qualType().bind(PointerType)))),
      hasTemplateArgument(
          1,
          templateArgument(refersToType(
              qualType(hasDeclaration(classTemplateSpecializationDecl(
                  hasName("::std::default_delete"), templateArgumentCountIs(1),
                  hasTemplateArgument(
                      0, templateArgument(refersToType(
                             qualType(equalsBoundNode(PointerType))))))))))))));
}

} // namespace modernize
} // namespace tidy
} // namespace clang
