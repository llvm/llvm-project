namespace outer {

template<typename T>
struct Foo {
  T t;
};

template <typename T, unsigned U, template<typename> class W>
struct Baz { };

template <typename... T>
struct Qux { };

namespace inner {

struct Bar {
  Bar(outer::Foo<bool>* foo) { }

  typedef int FooType;
  int *p;
  int *f(int *p, char *x, FooType z) {
    const FooType w = z;
    return p + z;
  }
  typedef double OtherType;
  typedef int ArrayType[5];
  Baz<int, 1, Foo> baz;
  Qux<int, char*, Foo<int>> qux;
};

}
}

template <typename T>
T tbar(int);

template <typename T>
T tbar(int[5]);

template <typename T, int size>
T tbar(int[size]);

void foo(int i, int incomplete_array[]) { int variable_array[i]; }

struct Blob {
  int i;
  int j;
};
int Blob::*member_pointer;



auto autoI = 0;
auto autoTbar = tbar<int>(0);
auto autoBlob = new Blob();
auto autoFunction(){return int();}
decltype(auto) autoInt = 5;

// RUN: c-index-test -test-print-type %s -std=c++14 | FileCheck %s
// CHECK: Namespace=outer:1:11 (Definition) [type=] [typekind=Invalid] [isPOD=0]
// CHECK: ClassTemplate=Foo:4:8 (Definition) [type=] [typekind=Invalid] [isPOD=0]
// CHECK: TemplateTypeParameter=T:3:19 (Definition) [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: FieldDecl=t:5:5 (Definition) [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: TypeRef=T:3:19 [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: ClassTemplate=Baz:9:8 (Definition) [type=] [typekind=Invalid] [isPOD=0]
// CHECK: TemplateTypeParameter=T:8:20 (Definition) [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: NonTypeTemplateParameter=U:8:32 (Definition) [type=unsigned int] [typekind=UInt] [isPOD=1]
// CHECK: TemplateTemplateParameter=W:8:60 (Definition) [type=] [typekind=Invalid] [isPOD=0]
// CHECK: Namespace=inner:14:11 (Definition) [type=] [typekind=Invalid] [isPOD=0]
// CHECK: StructDecl=Bar:16:8 (Definition) [type=outer::inner::Bar] [typekind=Record] [isPOD=0] [nbFields=3]
// CHECK: CXXConstructor=Bar:17:3 (Definition) [type=void (outer::Foo<bool> *){{.*}}] [typekind=FunctionProto] [canonicaltype=void (outer::Foo<bool> *){{.*}}] [canonicaltypekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [args= [outer::Foo<bool> *] [Pointer]] [isPOD=0]
// CHECK: ParmDecl=foo:17:25 (Definition) [type=outer::Foo<bool> *] [typekind=Pointer] [canonicaltype=outer::Foo<bool> *] [canonicaltypekind=Pointer] [isPOD=1] [pointeetype=outer::Foo<bool>] [pointeekind=Unexposed]
// CHECK: NamespaceRef=outer:1:11 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: TemplateRef=Foo:4:8 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: CompoundStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: TypedefDecl=FooType:19:15 (Definition) [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: FieldDecl=p:20:8 (Definition) [type=int *] [typekind=Pointer] [isPOD=1] [pointeetype=int] [pointeekind=Int]
// CHECK: CXXMethod=f:21:8 (Definition) [type=int *(int *, char *, FooType){{.*}}] [typekind=FunctionProto] [canonicaltype=int *(int *, char *, int){{.*}}] [canonicaltypekind=FunctionProto] [resulttype=int *] [resulttypekind=Pointer] [args= [int *] [Pointer] [char *] [Pointer] [FooType] [Typedef]] [isPOD=0]
// CHECK: ParmDecl=p:21:15 (Definition) [type=int *] [typekind=Pointer] [isPOD=1] [pointeetype=int] [pointeekind=Int]
// CHECK: ParmDecl=x:21:24 (Definition) [type=char *] [typekind=Pointer] [isPOD=1] [pointeetype=char] [pointeekind=Char_{{[US]}}]
// CHECK: ParmDecl=z:21:35 (Definition) [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: TypeRef=FooType:19:15 [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: CompoundStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: DeclStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: VarDecl=w:22:19 (Definition) [type=const FooType] [typekind=Typedef] const [canonicaltype=const int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: TypeRef=FooType:19:15 [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: UnexposedExpr=z:21:35 [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: DeclRefExpr=z:21:35 [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: ReturnStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: BinaryOperator= [type=int *] [typekind=Pointer] [isPOD=1] [pointeetype=int] [pointeekind=Int]
// CHECK: UnexposedExpr=p:21:15 [type=int *] [typekind=Pointer] [isPOD=1] [pointeetype=int] [pointeekind=Int]
// CHECK: DeclRefExpr=p:21:15 [type=int *] [typekind=Pointer] [isPOD=1] [pointeetype=int] [pointeekind=Int]
// CHECK: UnexposedExpr=z:21:35 [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: DeclRefExpr=z:21:35 [type=FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: TypedefDecl=OtherType:25:18 (Definition) [type=OtherType] [typekind=Typedef] [canonicaltype=double] [canonicaltypekind=Double] [isPOD=1]
// CHECK: TypedefDecl=ArrayType:26:15 (Definition) [type=ArrayType] [typekind=Typedef] [canonicaltype=int [5]] [canonicaltypekind=ConstantArray] [isPOD=1]
// CHECK: IntegerLiteral= [type=int] [typekind=Int] [isPOD=1]
// CHECK: FieldDecl=baz:27:20 (Definition) [type=Baz<int, 1, Foo>] [typekind=Unexposed] [canonicaltype=outer::Baz<int, 1, Foo>] [canonicaltypekind=Record] [templateargs/3= [type=int] [typekind=Int]] [isPOD=1]
// CHECK: TemplateRef=Baz:9:8 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: IntegerLiteral= [type=int] [typekind=Int] [isPOD=1]
// CHECK: TemplateRef=Foo:4:8 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: FieldDecl=qux:28:29 (Definition) [type=Qux<int, char *, Foo<int> >] [typekind=Unexposed] [canonicaltype=outer::Qux<int, char *, outer::Foo<int> >] [canonicaltypekind=Record] [templateargs/1=] [isPOD=1]
// CHECK: TemplateRef=Qux:12:8 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: TemplateRef=Foo:4:8 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: FunctionTemplate=tbar:35:3 [type=T (int)] [typekind=FunctionProto] [canonicaltype=type-parameter-0-0 (int)] [canonicaltypekind=FunctionProto] [resulttype=T] [resulttypekind=Unexposed] [isPOD=0]
// CHECK: TemplateTypeParameter=T:34:20 (Definition) [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: TypeRef=T:34:20 [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: ParmDecl=:35:11 (Definition) [type=int] [typekind=Int] [isPOD=1]
// CHECK: FunctionTemplate=tbar:38:3 [type=T (int *)] [typekind=FunctionProto] [canonicaltype=type-parameter-0-0 (int *)] [canonicaltypekind=FunctionProto] [resulttype=T] [resulttypekind=Unexposed] [isPOD=0]
// CHECK: TemplateTypeParameter=T:37:20 (Definition) [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: TypeRef=T:37:20 [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: ParmDecl=:38:11 (Definition) [type=int [5]] [typekind=ConstantArray] [isPOD=1]
// CHECK: IntegerLiteral= [type=int] [typekind=Int] [isPOD=1]
// CHECK: FunctionTemplate=tbar:41:3 [type=T (int *)] [typekind=FunctionProto] [canonicaltype=type-parameter-0-0 (int *)] [canonicaltypekind=FunctionProto] [resulttype=T] [resulttypekind=Unexposed] [isPOD=0]
// CHECK: TemplateTypeParameter=T:40:20 (Definition) [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: NonTypeTemplateParameter=size:40:27 (Definition) [type=int] [typekind=Int] [isPOD=1]
// CHECK: TypeRef=T:40:20 [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: ParmDecl=:41:11 (Definition) [type=int [size]] [typekind=DependentSizedArray] [isPOD=0]
// CHECK: DeclRefExpr=size:40:27 [type=int] [typekind=Int] [isPOD=1]
// CHECK: FunctionDecl=foo:43:6 (Definition) [type=void (int, int *)] [typekind=FunctionProto] [canonicaltype=void (int, int *)] [canonicaltypekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [args= [int] [Int] [int []] [IncompleteArray]] [isPOD=0]
// CHECK: ParmDecl=i:43:14 (Definition) [type=int] [typekind=Int] [isPOD=1]
// CHECK: ParmDecl=incomplete_array:43:21 (Definition) [type=int []] [typekind=IncompleteArray] [isPOD=1]
// CHECK: CompoundStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: DeclStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: VarDecl=variable_array:43:47 (Definition) [type=int [i]] [typekind=VariableArray] [isPOD=1]
// CHECK: DeclRefExpr=i:43:14 [type=int] [typekind=Int] [isPOD=1]
// CHECK: StructDecl=Blob:45:8 (Definition) [type=Blob] [typekind=Record] [isPOD=1] [nbFields=2]
// CHECK: FieldDecl=i:46:7 (Definition) [type=int] [typekind=Int] [isPOD=1]
// CHECK: VarDecl=member_pointer:49:12 (Definition) [type=int Blob::*] [typekind=MemberPointer] [isPOD=1]
// CHECK: VarDecl=autoI:53:6 (Definition) [type=int] [typekind=Auto] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: IntegerLiteral= [type=int] [typekind=Int] [isPOD=1]
// CHECK: VarDecl=autoTbar:54:6 (Definition) [type=int] [typekind=Auto] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: CallExpr=tbar:35:3 [type=int] [typekind=Unexposed] [canonicaltype=int] [canonicaltypekind=Int] [args= [int] [Int]] [isPOD=1]
// CHECK: UnexposedExpr=tbar:35:3 [type=int (*)(int)] [typekind=Pointer] [canonicaltype=int (*)(int)] [canonicaltypekind=Pointer] [isPOD=1] [pointeetype=int (int)] [pointeekind=FunctionProto]
// CHECK: DeclRefExpr=tbar:35:3 RefName=[54:17 - 54:21] RefName=[54:21 - 54:26] [type=int (int)] [typekind=FunctionProto] [canonicaltype=int (int)] [canonicaltypekind=FunctionProto] [isPOD=0]
// CHECK: IntegerLiteral= [type=int] [typekind=Int] [isPOD=1]
// CHECK: VarDecl=autoBlob:55:6 (Definition) [type=Blob *] [typekind=Auto] [canonicaltype=Blob *] [canonicaltypekind=Pointer] [isPOD=1]
// CHECK: CXXNewExpr= [type=Blob *] [typekind=Pointer] [isPOD=1] [pointeetype=Blob] [pointeekind=Record]
// CHECK: TypeRef=struct Blob:45:8 [type=Blob] [typekind=Record] [isPOD=1] [nbFields=2]
// CHECK: CallExpr=Blob:45:8 [type=Blob] [typekind=Record] [isPOD=1] [nbFields=2]
// CHECK: FunctionDecl=autoFunction:56:6 (Definition) [type=int ()] [typekind=FunctionProto] [canonicaltype=int ()] [canonicaltypekind=FunctionProto] [resulttype=int] [resulttypekind=Auto] [isPOD=0]
// CHECK: CompoundStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: ReturnStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: UnexposedExpr= [type=int] [typekind=Int] [isPOD=1]
// CHECK: VarDecl=autoInt:57:16 (Definition) [type=int] [typekind=Auto] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: IntegerLiteral= [type=int] [typekind=Int] [isPOD=1]
