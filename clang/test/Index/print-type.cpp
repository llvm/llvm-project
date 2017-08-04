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
  using AliasType = double;
  int *p;
  int *f(int *p, char *x, FooType z) {
    const FooType w = z;
    return p + z;
  }
  typedef double OtherType;
  typedef int ArrayType[5];
  Baz<int, 1, Foo> baz;
  Qux<int, char*, Foo<int>, FooType> qux;
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

namespace NS { struct Type{}; } NS::Type elaboratedNamespaceType(const NS::Type t);

auto autoI = 0;
auto autoTbar = tbar<int>(0);
auto autoBlob = new Blob();
auto autoFunction(){return int();}
decltype(auto) autoInt = 5;

template <typename T>
using TypeAlias = outer::Qux<T>;

struct TypeAliasUser { TypeAlias<int> foo; };

template<typename T>
struct Specialization {};

template<>
struct Specialization<int>;

Specialization<Specialization<bool>& > templRefParam;
auto autoTemplRefParam = templRefParam;

template<typename T> struct A {};
template<typename T> using C = T;
using baz = C<A<void>>;

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
// CHECK: CXXConstructor=Bar:17:3 (Definition) (converting constructor) [type=void (outer::Foo<bool> *){{.*}}] [typekind=FunctionProto] [canonicaltype=void (outer::Foo<bool> *){{.*}}] [canonicaltypekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [args= [outer::Foo<bool> *] [Pointer]] [isPOD=0]
// CHECK: ParmDecl=foo:17:25 (Definition) [type=outer::Foo<bool> *] [typekind=Pointer] [canonicaltype=outer::Foo<bool> *] [canonicaltypekind=Pointer] [isPOD=1] [pointeetype=outer::Foo<bool>] [pointeekind=Elaborated]
// CHECK: NamespaceRef=outer:1:11 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: TemplateRef=Foo:4:8 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: CompoundStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: TypedefDecl=FooType:19:15 (Definition) [type=outer::inner::Bar::FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: TypeAliasDecl=AliasType:20:9 (Definition) [type=outer::inner::Bar::AliasType] [typekind=Typedef] [canonicaltype=double] [canonicaltypekind=Double] [isPOD=1]
// CHECK: FieldDecl=p:21:8 (Definition) [type=int *] [typekind=Pointer] [isPOD=1] [pointeetype=int] [pointeekind=Int]
// CHECK: CXXMethod=f:22:8 (Definition) [type=int *(int *, char *, outer::inner::Bar::FooType){{.*}}] [typekind=FunctionProto] [canonicaltype=int *(int *, char *, int){{.*}}] [canonicaltypekind=FunctionProto] [resulttype=int *] [resulttypekind=Pointer] [args= [int *] [Pointer] [char *] [Pointer] [outer::inner::Bar::FooType] [Typedef]] [isPOD=0]
// CHECK: ParmDecl=p:22:15 (Definition) [type=int *] [typekind=Pointer] [isPOD=1] [pointeetype=int] [pointeekind=Int]
// CHECK: ParmDecl=x:22:24 (Definition) [type=char *] [typekind=Pointer] [isPOD=1] [pointeetype=char] [pointeekind=Char_{{[US]}}]
// CHECK: ParmDecl=z:22:35 (Definition) [type=outer::inner::Bar::FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: TypeRef=outer::inner::Bar::FooType:19:15 [type=outer::inner::Bar::FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: CompoundStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: DeclStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: VarDecl=w:23:19 (Definition) [type=const outer::inner::Bar::FooType] [typekind=Typedef] const [canonicaltype=const int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: TypeRef=outer::inner::Bar::FooType:19:15 [type=outer::inner::Bar::FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: UnexposedExpr=z:22:35 [type=outer::inner::Bar::FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: DeclRefExpr=z:22:35 [type=outer::inner::Bar::FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: ReturnStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: BinaryOperator= [type=int *] [typekind=Pointer] [isPOD=1] [pointeetype=int] [pointeekind=Int]
// CHECK: UnexposedExpr=p:22:15 [type=int *] [typekind=Pointer] [isPOD=1] [pointeetype=int] [pointeekind=Int]
// CHECK: DeclRefExpr=p:22:15 [type=int *] [typekind=Pointer] [isPOD=1] [pointeetype=int] [pointeekind=Int]
// CHECK: UnexposedExpr=z:22:35 [type=outer::inner::Bar::FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: DeclRefExpr=z:22:35 [type=outer::inner::Bar::FooType] [typekind=Typedef] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: TypedefDecl=OtherType:26:18 (Definition) [type=outer::inner::Bar::OtherType] [typekind=Typedef] [canonicaltype=double] [canonicaltypekind=Double] [isPOD=1]
// CHECK: TypedefDecl=ArrayType:27:15 (Definition) [type=outer::inner::Bar::ArrayType] [typekind=Typedef] [canonicaltype=int [5]] [canonicaltypekind=ConstantArray] [isPOD=1]
// CHECK: IntegerLiteral= [type=int] [typekind=Int] [isPOD=1]
// CHECK: FieldDecl=baz:28:20 (Definition) [type=Baz<int, 1, Foo>] [typekind=Unexposed] [templateargs/3= [type=int] [typekind=Int]] [canonicaltype=outer::Baz<int, 1, Foo>] [canonicaltypekind=Record] [canonicaltemplateargs/3= [type=int] [typekind=Int]] [isPOD=1]
// CHECK: TemplateRef=Baz:9:8 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: IntegerLiteral= [type=int] [typekind=Int] [isPOD=1]
// CHECK: TemplateRef=Foo:4:8 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: FieldDecl=qux:29:38 (Definition) [type=Qux<int, char *, Foo<int>, outer::inner::Bar::FooType>] [typekind=Unexposed] [templateargs/4= [type=int] [typekind=Int] [type=char *] [typekind=Pointer] [type=Foo<int>] [typekind=Unexposed] [type=outer::inner::Bar::FooType] [typekind=Typedef]] [canonicaltype=outer::Qux<int, char *, outer::Foo<int>, int>] [canonicaltypekind=Record] [canonicaltemplateargs/4= [type=int] [typekind=Int] [type=char *] [typekind=Pointer] [type=outer::Foo<int>] [typekind=Record] [type=int] [typekind=Int]] [isPOD=1]
// CHECK: TemplateRef=Qux:12:8 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: TemplateRef=Foo:4:8 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: FunctionTemplate=tbar:36:3 [type=T (int)] [typekind=FunctionProto] [canonicaltype=type-parameter-0-0 (int)] [canonicaltypekind=FunctionProto] [resulttype=T] [resulttypekind=Unexposed] [isPOD=0]
// CHECK: TemplateTypeParameter=T:35:20 (Definition) [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: TypeRef=T:35:20 [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: ParmDecl=:36:11 (Definition) [type=int] [typekind=Int] [isPOD=1]
// CHECK: FunctionTemplate=tbar:39:3 [type=T (int *)] [typekind=FunctionProto] [canonicaltype=type-parameter-0-0 (int *)] [canonicaltypekind=FunctionProto] [resulttype=T] [resulttypekind=Unexposed] [isPOD=0]
// CHECK: TemplateTypeParameter=T:38:20 (Definition) [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: TypeRef=T:38:20 [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: ParmDecl=:39:11 (Definition) [type=int [5]] [typekind=ConstantArray] [isPOD=1]
// CHECK: IntegerLiteral= [type=int] [typekind=Int] [isPOD=1]
// CHECK: FunctionTemplate=tbar:42:3 [type=T (int *)] [typekind=FunctionProto] [canonicaltype=type-parameter-0-0 (int *)] [canonicaltypekind=FunctionProto] [resulttype=T] [resulttypekind=Unexposed] [isPOD=0]
// CHECK: TemplateTypeParameter=T:41:20 (Definition) [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: NonTypeTemplateParameter=size:41:27 (Definition) [type=int] [typekind=Int] [isPOD=1]
// CHECK: TypeRef=T:41:20 [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: ParmDecl=:42:11 (Definition) [type=int [size]] [typekind=DependentSizedArray] [isPOD=0]
// CHECK: DeclRefExpr=size:41:27 [type=int] [typekind=Int] [isPOD=1]
// CHECK: FunctionDecl=foo:44:6 (Definition) [type=void (int, int *)] [typekind=FunctionProto] [canonicaltype=void (int, int *)] [canonicaltypekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [args= [int] [Int] [int []] [IncompleteArray]] [isPOD=0]
// CHECK: ParmDecl=i:44:14 (Definition) [type=int] [typekind=Int] [isPOD=1]
// CHECK: ParmDecl=incomplete_array:44:21 (Definition) [type=int []] [typekind=IncompleteArray] [isPOD=1]
// CHECK: CompoundStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: DeclStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: VarDecl=variable_array:44:47 (Definition) [type=int [i]] [typekind=VariableArray] [isPOD=1]
// CHECK: DeclRefExpr=i:44:14 [type=int] [typekind=Int] [isPOD=1]
// CHECK: StructDecl=Blob:46:8 (Definition) [type=Blob] [typekind=Record] [isPOD=1] [nbFields=2]
// CHECK: FieldDecl=i:47:7 (Definition) [type=int] [typekind=Int] [isPOD=1]
// CHECK: VarDecl=member_pointer:50:12 (Definition) [type=int Blob::*] [typekind=MemberPointer] [isPOD=1]
// CHECK: FunctionDecl=elaboratedNamespaceType:52:42 [type=NS::Type (const NS::Type)] [typekind=FunctionProto] [canonicaltype=NS::Type (NS::Type)] [canonicaltypekind=FunctionProto] [resulttype=NS::Type] [resulttypekind=Elaborated] [args= [const NS::Type] [Elaborated]] [isPOD=0]
// CHECK: NamespaceRef=NS:52:11 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: TypeRef=struct NS::Type:52:23 [type=NS::Type] [typekind=Record] [isPOD=1]
// CHECK: ParmDecl=t:52:81 (Definition) [type=const NS::Type] [typekind=Elaborated] const [canonicaltype=const NS::Type] [canonicaltypekind=Record] [isPOD=1]
// CHECK: VarDecl=autoI:54:6 (Definition) [type=int] [typekind=Auto] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: IntegerLiteral= [type=int] [typekind=Int] [isPOD=1]
// CHECK: VarDecl=autoTbar:55:6 (Definition) [type=int] [typekind=Auto] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: CallExpr=tbar:36:3 [type=int] [typekind=Unexposed] [canonicaltype=int] [canonicaltypekind=Int] [args= [int] [Int]] [isPOD=1]
// CHECK: UnexposedExpr=tbar:36:3 [type=int (*)(int)] [typekind=Pointer] [canonicaltype=int (*)(int)] [canonicaltypekind=Pointer] [isPOD=1] [pointeetype=int (int)] [pointeekind=FunctionProto]
// CHECK: DeclRefExpr=tbar:36:3 RefName=[55:17 - 55:21] RefName=[55:21 - 55:26] [type=int (int)] [typekind=FunctionProto] [canonicaltype=int (int)] [canonicaltypekind=FunctionProto] [isPOD=0]
// CHECK: IntegerLiteral= [type=int] [typekind=Int] [isPOD=1]
// CHECK: VarDecl=autoBlob:56:6 (Definition) [type=Blob *] [typekind=Auto] [canonicaltype=Blob *] [canonicaltypekind=Pointer] [isPOD=1]
// CHECK: CXXNewExpr= [type=Blob *] [typekind=Pointer] [isPOD=1] [pointeetype=Blob] [pointeekind=Record]
// CHECK: TypeRef=struct Blob:46:8 [type=Blob] [typekind=Record] [isPOD=1] [nbFields=2]
// CHECK: CallExpr=Blob:46:8 [type=Blob] [typekind=Record] [isPOD=1] [nbFields=2]
// CHECK: FunctionDecl=autoFunction:57:6 (Definition) [type=int ()] [typekind=FunctionProto] [canonicaltype=int ()] [canonicaltypekind=FunctionProto] [resulttype=int] [resulttypekind=Auto] [isPOD=0]
// CHECK: CompoundStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: ReturnStmt= [type=] [typekind=Invalid] [isPOD=0]
// CHECK: UnexposedExpr= [type=int] [typekind=Int] [isPOD=1]
// CHECK: VarDecl=autoInt:58:16 (Definition) [type=int] [typekind=Auto] [canonicaltype=int] [canonicaltypekind=Int] [isPOD=1]
// CHECK: IntegerLiteral= [type=int] [typekind=Int] [isPOD=1]
// CHECK: TypeAliasTemplateDecl=TypeAlias:61:1 (Definition) [type=] [typekind=Invalid] [isPOD=0]
// CHECK: TemplateTypeParameter=T:60:20 (Definition) [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: FieldDecl=foo:63:39 (Definition) [type=TypeAlias<int>] [typekind=Unexposed] [templateargs/1= [type=int] [typekind=Int]] [canonicaltype=outer::Qux<int>] [canonicaltypekind=Record] [canonicaltemplateargs/1= [type=int] [typekind=Int]] [isPOD=1]
// CHECK: TemplateRef=TypeAlias:61:1 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: ClassTemplate=Specialization:66:8 (Definition) [type=] [typekind=Invalid] [isPOD=0]
// CHECK: TemplateTypeParameter=T:65:19 (Definition) [type=T] [typekind=Unexposed] [canonicaltype=type-parameter-0-0] [canonicaltypekind=Unexposed] [isPOD=0]
// CHECK: StructDecl=Specialization:69:8 [Specialization of Specialization:66:8] [type=Specialization<int>] [typekind=Record] [templateargs/1= [type=int] [typekind=Int]] [isPOD=0]
// CHECK: VarDecl=templRefParam:71:40 (Definition) [type=Specialization<Specialization<bool> &>] [typekind=Unexposed] [templateargs/1= [type=Specialization<bool> &] [typekind=LValueReference]] [canonicaltype=Specialization<Specialization<bool> &>] [canonicaltypekind=Record] [canonicaltemplateargs/1= [type=Specialization<bool> &] [typekind=LValueReference]] [isPOD=1]
// CHECK: TemplateRef=Specialization:66:8 [type=] [typekind=Invalid] [isPOD=0]
// CHECK: CallExpr=Specialization:66:8 [type=Specialization<Specialization<bool> &>] [typekind=Unexposed] [templateargs/1= [type=Specialization<bool> &] [typekind=LValueReference]] [canonicaltype=Specialization<Specialization<bool> &>] [canonicaltypekind=Record] [canonicaltemplateargs/1= [type=Specialization<bool> &] [typekind=LValueReference]] [isPOD=1]
// CHECK: VarDecl=autoTemplRefParam:72:6 (Definition) [type=Specialization<Specialization<bool> &>] [typekind=Auto] [templateargs/1= [type=Specialization<bool> &] [typekind=LValueReference]] [canonicaltype=Specialization<Specialization<bool> &>] [canonicaltypekind=Record] [canonicaltemplateargs/1= [type=Specialization<bool> &] [typekind=LValueReference]] [isPOD=1]
// CHECK: UnexposedExpr=templRefParam:71:40 [type=const Specialization<Specialization<bool> &>] [typekind=Unexposed] const [templateargs/1= [type=Specialization<bool> &] [typekind=LValueReference]] [canonicaltype=const Specialization<Specialization<bool> &>] [canonicaltypekind=Record] [canonicaltemplateargs/1= [type=Specialization<bool> &] [typekind=LValueReference]] [isPOD=1]
// CHECK: DeclRefExpr=templRefParam:71:40 [type=Specialization<Specialization<bool> &>] [typekind=Unexposed] [templateargs/1= [type=Specialization<bool> &] [typekind=LValueReference]] [canonicaltype=Specialization<Specialization<bool> &>] [canonicaltypekind=Record] [canonicaltemplateargs/1= [type=Specialization<bool> &] [typekind=LValueReference]] [isPOD=1]
// CHECK: TypeAliasDecl=baz:76:7 (Definition) [type=baz] [typekind=Typedef] [templateargs/1= [type=A<void>] [typekind=Unexposed]] [canonicaltype=A<void>] [canonicaltypekind=Record] [canonicaltemplateargs/1= [type=void] [typekind=Void]] [isPOD=0]
