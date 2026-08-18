//===-- clang-doc/ClangDocTest.h ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANG_DOC_CLANGDOCTEST_H
#define LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANG_DOC_CLANGDOCTEST_H

#include "ClangDocTest.h"
#include "Representation.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "gtest/gtest.h"

namespace clang {
namespace doc {

using EmittedInfoList = std::vector<Info *>;

static const SymbolID EmptySID = SymbolID();
static const SymbolID NonEmptySID =
    SymbolID{{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

NamespaceInfo *InfoAsNamespace(Info *I);
RecordInfo *InfoAsRecord(Info *I);
FunctionInfo *InfoAsFunction(Info *I);
EnumInfo *InfoAsEnum(Info *I);
TypedefInfo *InfoAsTypedef(Info *I);

void CheckCommentInfo(ArrayRef<CommentInfo> Expected,
                      ArrayRef<CommentInfo> Actual);
void CheckCommentInfo(const DocList<CommentInfo> &Expected,
                      const DocList<CommentInfo> &Actual);
void CheckReference(const Reference &Expected, const Reference &Actual);
void CheckTypeInfo(const TypeInfo *Expected, const TypeInfo *Actual);
void CheckFieldTypeInfo(const FieldTypeInfo *Expected,
                        const FieldTypeInfo *Actual);
void CheckMemberTypeInfo(const MemberTypeInfo *Expected,
                         const MemberTypeInfo *Actual);

// This function explicitly does not check USRs, as that may change and it would
// be better to not rely on its implementation.
void CheckBaseInfo(const Info *Expected, const Info *Actual);
void CheckSymbolInfo(const SymbolInfo *Expected, const SymbolInfo *Actual);
void CheckFunctionInfo(const FunctionInfo *Expected,
                       const FunctionInfo *Actual);
void CheckEnumInfo(const EnumInfo *Expected, const EnumInfo *Actual);
void CheckTypedefInfo(const TypedefInfo *Expected, const TypedefInfo *Actual);
void CheckNamespaceInfo(const NamespaceInfo *Expected,
                        const NamespaceInfo *Actual);
void CheckRecordInfo(const RecordInfo *Expected, const RecordInfo *Actual);
void CheckBaseRecordInfo(const BaseRecordInfo *Expected,
                         const BaseRecordInfo *Actual);

void CheckIndex(const Index &Expected, const Index &Actual);

class ClangDocContextTest : public ::testing::Test {
protected:
  ClangDocContextTest();
  ~ClangDocContextTest() override;

  ClangDocContext
  getClangDocContext(std::vector<std::string> UserStylesheets = {},
                     StringRef RepositoryUrl = "",
                     StringRef RepositoryLinePrefix = "", StringRef Base = "");

  DiagnosticOptions DiagOpts;
  llvm::IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  DiagnosticsEngine Diags;
};

} // namespace doc
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_UNITTESTS_CLANG_DOC_CLANGDOCTEST_H
