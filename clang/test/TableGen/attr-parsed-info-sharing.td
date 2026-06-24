// RUN: clang-tblgen -gen-clang-attr-parsed-attr-impl -I%p/../../include %s -o %t
// RUN: FileCheck %s < %t
// RUN: grep "diag::err_decl_attribute_invalid_on_stmt" %t | count 1
// RUN: grep "diag::err_attribute_invalid_on_decl" %t | count 1

include "clang/Basic/Attr.td"

def TestDeclOnly : InheritableAttr {
  let Spellings = [Clang<"test_decl_only">];
  let Subjects = SubjectList<[Function]>;
  let Documentation = [Undocumented];
}

def TestStmtOnly : StmtAttr {
  let Spellings = [Clang<"test_stmt_only">];
  let Subjects = SubjectList<[NullStmt], ErrorDiag, "empty statements">;
  let Documentation = [Undocumented];
}

def TestDeclAndStmt : DeclOrStmtAttr {
  let Spellings = [Clang<"test_decl_and_stmt">];
  let Subjects =
      SubjectList<[Function, NullStmt], ErrorDiag,
                  "functions and empty statements">;
  let Documentation = [Undocumented];
}

// CHECK-LABEL: struct DeclOnlyParsedAttrInfo : ParsedAttrInfo {
// CHECK: bool diagAppertainsToStmt(Sema &S, const ParsedAttr &AL,
// CHECK: S.Diag(AL.getLoc(), diag::err_decl_attribute_invalid_on_stmt)
// CHECK: static_assert(sizeof(DeclOnlyParsedAttrInfo) == sizeof(ParsedAttrInfo));

// CHECK-LABEL: struct StmtOnlyParsedAttrInfo : ParsedAttrInfo {
// CHECK: bool diagAppertainsToDecl(Sema &S, const ParsedAttr &AL,
// CHECK: S.Diag(AL.getLoc(), diag::err_attribute_invalid_on_decl)
// CHECK: static_assert(sizeof(StmtOnlyParsedAttrInfo) == sizeof(ParsedAttrInfo));

// CHECK-LABEL: struct ParsedAttrInfoTestDeclAndStmt final : public ParsedAttrInfo {
// CHECK: constexpr ParsedAttrInfoTestDeclAndStmt() : ParsedAttrInfo(
// CHECK: bool diagAppertainsToDecl(
// CHECK: bool diagAppertainsToStmt(

// CHECK-LABEL: struct ParsedAttrInfoTestDeclOnly final : public DeclOnlyParsedAttrInfo {
// CHECK: constexpr ParsedAttrInfoTestDeclOnly() : DeclOnlyParsedAttrInfo(
// CHECK: bool diagAppertainsToDecl(
// CHECK-NOT: bool diagAppertainsToStmt(
// CHECK: static const ParsedAttrInfoTestDeclOnly Instance;

// CHECK-LABEL: struct ParsedAttrInfoTestStmtOnly final : public StmtOnlyParsedAttrInfo {
// CHECK: constexpr ParsedAttrInfoTestStmtOnly() : StmtOnlyParsedAttrInfo(
// CHECK: bool diagAppertainsToStmt(
// CHECK-NOT: bool diagAppertainsToDecl(
// CHECK: static const ParsedAttrInfoTestStmtOnly Instance;
