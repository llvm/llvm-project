// RUN: c-index-test -test-print-type %s | FileCheck --check-prefix=CHECK-TYPE %s
// RUN: c-index-test -test-print-type-declaration %s | FileCheck --check-prefix=CHECK-TYPEDECL %s

// CHECK-TYPE: VarDecl=VariableWithParentheses:
// CHECK-TYPE-SAME: [type=int] [typekind=Int]
// CHECK-TYPE-NOT: canonicaltype
// CHECK-TYPE-SAME: isPOD
extern int (VariableWithParentheses);

typedef int MyTypedef;
// CHECK-TYPE: VarDecl=VariableWithParentheses2:
// CHECK-TYPE-SAME: [type=MyTypedef] [typekind=Typedef]
// CHECK-TYPE-SAME: [canonicaltype=int] [canonicaltypekind=Int]
// CHECK-TYPEDECL: VarDecl=VariableWithParentheses2
// CHECK-TYPEDECL-SAME: [typedeclaration=MyTypedef] [typekind=Typedef]
extern MyTypedef (VariableWithParentheses2);
