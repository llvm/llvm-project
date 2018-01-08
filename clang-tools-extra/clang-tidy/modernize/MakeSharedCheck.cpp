//===--- MakeSharedCheck.cpp - clang-tidy----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MakeSharedCheck.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace modernize {

MakeSharedCheck::MakeSharedCheck(StringRef Name, ClangTidyContext *Context)
    : MakeSmartPtrCheck(Name, Context, "std::make_shared") {}

MakeSharedCheck::SmartPtrTypeMatcher
MakeSharedCheck::getSmartPointerTypeMatcher() const {
  return qualType(hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(classTemplateSpecializationDecl(
          hasName("::std::shared_ptr"), templateArgumentCountIs(1),
          hasTemplateArgument(0, templateArgument(refersToType(
                                     qualType().bind(PointerType)))))))));
}

} // namespace modernize
} // namespace tidy
} // namespace clang
