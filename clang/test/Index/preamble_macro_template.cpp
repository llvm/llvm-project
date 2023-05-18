template void foo(int *);

int main() { }

// RUN: c-index-test -write-pch %t.pch -fno-delayed-template-parsing -x c++-header %S/Inputs/preamble_macro_template.h
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 5 local -fno-delayed-template-parsing -I %S/Inputs -include %t %s 2>&1 | tee %t.check.txt | FileCheck %s
// CHECK: preamble_macro_template.h:4:6: FunctionDecl=foo:4:6 (Definition) [Specialization of foo:4:6] [Template arg 0: kind: 1, type: int] Extent=[4:1 - 6:2]
// CHECK: preamble_macro_template.h:4:13: ParmDecl=p:4:13 (Definition) Extent=[4:10 - 4:14]
// CHECK: preamble_macro_template.h:4:16: CompoundStmt= Extent=[4:16 - 6:2]
// CHECK: preamble_macro_template.h:5:3: CStyleCastExpr= Extent=[5:3 - 5:27]
// CHECK: preamble_macro_template.h:5:9: CXXStaticCastExpr= Extent=[5:9 - 5:27]
// CHECK: preamble_macro_template.h:5:25: UnexposedExpr= Extent=[5:25 - 5:26]
// CHECK: preamble_macro_template.h:5:25: IntegerLiteral= Extent=[5:25 - 5:26]
// CHECK: preamble_macro_template.cpp:3:5: FunctionDecl=main:3:5 (Definition) Extent=[3:1 - 3:15]
// CHECK: preamble_macro_template.cpp:3:12: CompoundStmt= Extent=[3:12 - 3:15]
