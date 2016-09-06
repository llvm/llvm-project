typedef int FooType;
int *p;
int *f(int *p, char *x, FooType z, int arr[5], void (*fn)(int)) {
  fn(*p);
  const FooType w = z;
  return p + z + arr[3];
}
typedef double OtherType;
typedef int ArrayType[5];
int __attribute__((vector_size(16))) x;
typedef int __attribute__((vector_size(16))) int4_t;

int f2(int incompletearray[]);

enum Enum{i}; enum Enum elaboratedEnumType();
struct Struct{}; struct Struct elaboratedStructType();

// RUN: c-index-test -test-print-type %s | FileCheck %s
// CHECK: FunctionDecl=f:3:6 (Definition) [type=int *(int *, char *, FooType, int *, void (*)(int))] [typekind=FunctionProto] [canonicaltype=int *(int *, char *, int, int *, void (*)(int))] [canonicaltypekind=FunctionProto] [resulttype=int *] [resulttypekind=Pointer] [args= [int *] [Pointer] [char *] [Pointer] [FooType] [Typedef] [int [5]] [ConstantArray] [void (*)(int)] [Pointer]] [isPOD=0]
// CHECK: ParmDecl=p:3:13 (Definition) [type=int *] [typekind=Pointer] [isPOD=1] [pointeetype=int] [pointeekind=Int]
// CHECK: ParmDecl=x:3:22 (Definition) [type=char *] [typekind=Pointer] [isPOD=1] [pointeetype=char] [pointeekind=Char_{{[US]}}]
// CHECK: ParmDecl=z:3:33 (Definition) [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: TypeRef=FooType:1:13 [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: ParmDecl=arr:3:40 (Definition) [type=int [5]] [typekind=ConstantArray] [isPOD=1]
// CHECK: IntegerLiteral= [type=int] [typekind=Int] [isPOD=1]
// CHECK: ParmDecl=fn:3:55 (Definition) [type=void (*)(int)] [typekind=Pointer] [canonicaltype=void (*)(int)] [canonicaltypekind=Pointer] [isPOD=1] [pointeetype=void (int)] [pointeekind=Unexposed]
// CHECK: ParmDecl=:3:62 (Definition) [type=int] [typekind=Int] [isPOD=1]
// CHECK: CompoundStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: CallExpr=fn:3:55 [type=void] [typekind=Void] [args= [int] [Int]] [isPOD=0]
// CHECK: DeclRefExpr=fn:3:55 [type=void (*)(int)] [typekind=Pointer] [canonicaltype=void (*)(int)] [canonicaltypekind=Pointer] [isPOD=1] [pointeetype=void (int)] [pointeekind=Unexposed]
// CHECK: UnaryOperator= [type=int] [typekind=Int] [isPOD=1]
// CHECK: DeclRefExpr=p:3:13 [type=int *] [typekind=Pointer] [isPOD=1] [pointeetype=int] [pointeekind=Int]
// CHECK: DeclStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: VarDecl=w:5:17 (Definition) [type=const FooType] [typekind=Typedef] const [canonicaltype=const int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: TypeRef=FooType:1:13 [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: DeclRefExpr=z:3:33 [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: ReturnStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: BinaryOperator= [type=int *] [typekind=Pointer] [isPOD=1] [pointeetype=int] [pointeekind=Int]
// CHECK: BinaryOperator= [type=int *] [typekind=Pointer] [isPOD=1] [pointeetype=int] [pointeekind=Int]
// CHECK: DeclRefExpr=p:3:13 [type=int *] [typekind=Pointer] [isPOD=1] [pointeetype=int] [pointeekind=Int]
// CHECK: DeclRefExpr=z:3:33 [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: ArraySubscriptExpr= [type=int] [typekind=Int] [isPOD=1]
// CHECK: UnexposedExpr=arr:3:40 [type=int [5]] [typekind=ConstantArray] [isPOD=1]
// CHECK: IntegerLiteral= [type=int] [typekind=Int] [isPOD=1]
// CHECK: TypedefDecl=OtherType:8:16 (Definition) [type=OtherType] [typekind=Typedef] [canonicaltype=double] [canonicaltypekind=Double] [isPOD=1]
// CHECK: TypedefDecl=ArrayType:9:13 (Definition) [type=ArrayType] [typekind=Typedef] [canonicaltype=int [5]] [canonicaltypekind=ConstantArray] [isPOD=1]
// CHECK: IntegerLiteral= [type=int] [typekind=Int] [isPOD=1]
// CHECK: VarDecl=x:10:38 [type=__attribute__((__vector_size__(4 * sizeof(int)))) int] [typekind=Vector] [isPOD=1]
// CHECK: TypedefDecl=int4_t:11:46 (Definition) [type=int4_t] [typekind=Typedef] [canonicaltype=__attribute__((__vector_size__(4 * sizeof(int)))) int] [canonicaltypekind=Vector] [isPOD=1]
// CHECK: ParmDecl=incompletearray:13:12 (Definition) [type=int []] [typekind=IncompleteArray] [isPOD=1]
// CHECK: FunctionDecl=elaboratedEnumType:15:25 [type=enum Enum ()] [typekind=FunctionNoProto] [canonicaltype=enum Enum ()] [canonicaltypekind=FunctionNoProto] [resulttype=enum Enum] [resulttypekind=Elaborated] [isPOD=0]
// CHECK: TypeRef=enum Enum:15:6 [type=enum Enum] [typekind=Enum] [isPOD=1]
// CHECK: StructDecl=Struct:16:8 (Definition) [type=struct Struct] [typekind=Record] [isPOD=1]
// CHECK: FunctionDecl=elaboratedStructType:16:32 [type=struct Struct ()] [typekind=FunctionNoProto] [canonicaltype=struct Struct ()] [canonicaltypekind=FunctionNoProto] [resulttype=struct Struct] [resulttypekind=Elaborated] [isPOD=0]
// CHECK: TypeRef=struct Struct:16:8 [type=struct Struct] [typekind=Record] [isPOD=1]
