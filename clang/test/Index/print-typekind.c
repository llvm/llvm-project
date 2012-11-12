typedef int FooType;
int *p;
int *f(int *p, char *x, FooType z) {
  const FooType w = z;
  return p + z;
}
typedef double OtherType;
typedef int ArrayType[5];

// RUN: c-index-test -test-print-typekind %s | FileCheck %s
// CHECK: TypedefDecl=FooType:1:13 (Definition) typekind=Typedef [canonical=Int] [isPOD=1]
// CHECK: VarDecl=p:2:6 typekind=Pointer [isPOD=1]
// CHECK: FunctionDecl=f:3:6 (Definition) typekind=FunctionProto [canonical=FunctionProto] [result=Pointer] [args= Pointer Pointer Typedef] [isPOD=0]
// CHECK: ParmDecl=p:3:13 (Definition) typekind=Pointer [isPOD=1]
// CHECK: ParmDecl=x:3:22 (Definition) typekind=Pointer [isPOD=1]
// CHECK: ParmDecl=z:3:33 (Definition) typekind=Typedef [canonical=Int] [isPOD=1]
// CHECK: TypeRef=FooType:1:13 typekind=Typedef [canonical=Int] [isPOD=1]
// CHECK: CompoundStmt= typekind=Invalid [isPOD=0]
// CHECK: DeclStmt= typekind=Invalid [isPOD=0]
// CHECK: VarDecl=w:4:17 (Definition) typekind=Typedef const [canonical=Int] [isPOD=1]
// CHECK: TypeRef=FooType:1:13 typekind=Typedef [canonical=Int] [isPOD=1]
// CHECK: DeclRefExpr=z:3:33 typekind=Typedef [canonical=Int] [isPOD=1]
// CHECK: ReturnStmt= typekind=Invalid [isPOD=0]
// CHECK: BinaryOperator= typekind=Pointer [isPOD=1]
// CHECK: DeclRefExpr=p:3:13 typekind=Pointer [isPOD=1]
// CHECK: DeclRefExpr=z:3:33 typekind=Typedef [canonical=Int] [isPOD=1]
// CHECK: TypedefDecl=OtherType:7:16 (Definition) typekind=Typedef [canonical=Double] [isPOD=1]
// CHECK: TypedefDecl=ArrayType:8:13 (Definition) typekind=Typedef [canonical=ConstantArray] [isPOD=1]
