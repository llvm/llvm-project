// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-pc-linux -std=c++11 -Wno-deprecated-declarations -ast-dump -ast-dump-filter Test %s \
// RUN: | FileCheck --strict-whitespace %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-pc-linux -std=c++11 -Wno-deprecated-declarations -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -triple x86_64-pc-linux -std=c++11 -Wno-deprecated-declarations \
// RUN: -include-pch %t -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace %s

int TestLocation
__attribute__((unused));
// CHECK: VarDecl 0x{{.+}} <{{.+}}> col:5 TestLocation 'int'
// CHECK-NEXT: typeDetails: BuiltinType
// CHECK-NEXT: `-attrDetails: UnusedAttr 0x{{.+}} <{{.+}}> unused

int TestIndent
__attribute__((unused));
// CHECK: VarDecl 0x{{.+}} <{{.+}}> col:5 TestIndent 'int'
// CHECK-NEXT: typeDetails: BuiltinType
// CHECK-NEXT: `-attrDetails: UnusedAttr 0x{{.+}} <{{.+}}> unused

void TestAttributedStmt() {
  switch (1) {
  case 1:
    [[clang::fallthrough]];
  case 2:
    ;
  }
}
// CHECK:      FunctionDecl{{.*}}TestAttributedStmt
// CHECK-NEXT: `-CompoundStmt 0x{{.+}} <{{.+}}>
// CHECK-NEXT:   `-SwitchStmt 0x{{.+}} <{{.+}}>
// CHECK-NEXT:     |-IntegerLiteral 0x{{.+}} <{{.+}}> 'int' 1
// CHECK-NEXT:     `-CompoundStmt 0x{{.+}} <{{.+}}>
// CHECK-NEXT:       |-CaseStmt 0x{{.+}} <{{.+}}>
// CHECK-NEXT:       | |-ConstantExpr 0x{{.+}} <{{.+}}> 'int'
// CHECK-NEXT:       | | |-value: Int 1
// CHECK-NEXT:       | | `-IntegerLiteral 0x{{.+}} <{{.+}}> 'int' 1
// CHECK-NEXT:       | `-AttributedStmt 0x{{.+}} <{{.+}}>
// CHECK-NEXT:       |   |-attrDetails: FallThroughAttr 0x{{.+}} <{{.+}}>
// CHECK-NEXT:       |   `-NullStmt 0x{{.+}} <{{.+}}>
// CHECK-NEXT:       `-CaseStmt 0x{{.+}} <{{.+}}>
// CHECK-NEXT:         |-ConstantExpr 0x{{.+}} <{{.+}}> 'int'
// CHECK-NEXT:         | |-value: Int 2
// CHECK-NEXT:         | `-IntegerLiteral 0x{{.+}} <{{.+}}> 'int' 2
// CHECK-NEXT:         `-NullStmt 0x{{.+}} <{{.+}}>

[[clang::warn_unused_result]] int TestCXX11DeclAttr();
// CHECK:      FunctionDecl{{.*}}TestCXX11DeclAttr
// CHECK-NEXT:   attrDetails: WarnUnusedResultAttr

int TestAlignedNull __attribute__((aligned));
// CHECK:      VarDecl{{.*}}TestAlignedNull
// CHECK-NEXT:   typeDetails: BuiltinType
// CHECK-NEXT:   attrDetails: AlignedAttr {{.*}} aligned
// CHECK-NEXT:     <<<NULL>>>

int TestAlignedExpr __attribute__((aligned(4)));
// CHECK:      VarDecl{{.*}}TestAlignedExpr
// CHECK-NEXT: typeDetails: BuiltinType
// CHECK-NEXT:   attrDetails: AlignedAttr {{.*}} aligned
// CHECK-NEXT:     ConstantExpr
// CHECK-NEXT:       value: Int 4
// CHECK-NEXT:       IntegerLiteral

int TestEnum __attribute__((visibility("default")));
// CHECK:      VarDecl{{.*}}TestEnum
// CHECK-NEXT: typeDetails: BuiltinType
// CHECK-NEXT:   attrDetails: VisibilityAttr{{.*}} Default

class __attribute__((lockable)) Mutex {
} mu1, mu2;
int TestExpr __attribute__((guarded_by(mu1)));
// CHECK:      VarDecl{{.*}}TestExpr
// CHECK-NEXT: typeDetails: BuiltinType
// CHECK-NEXT:   attrDetails: GuardedByAttr
// CHECK-NEXT:     DeclRefExpr{{.*}}mu1

class Mutex TestVariadicExpr __attribute__((acquired_after(mu1, mu2)));
// CHECK:      VarDecl{{.*}}TestVariadicExpr
// CHECK:        attrDetails: AcquiredAfterAttr
// CHECK-NEXT:     DeclRefExpr{{.*}}mu1
// CHECK-NEXT:     DeclRefExpr{{.*}}mu2

void function1(void *) {
  int TestFunction __attribute__((cleanup(function1)));
}
// CHECK:      VarDecl{{.*}}TestFunction
// CHECK-NEXT: typeDetails: BuiltinType
// CHECK-NEXT:   attrDetails: CleanupAttr{{.*}} Function{{.*}}function1

void TestIdentifier(void *, int)
__attribute__((pointer_with_type_tag(ident1,1,2)));
// CHECK: FunctionDecl{{.*}}TestIdentifier
// CHECK: typeDetails: BuiltinType
// CHECK:   attrDetails: ArgumentWithTypeTagAttr{{.*}} pointer_with_type_tag ident1

void TestBool(void *, int)
__attribute__((pointer_with_type_tag(bool1,1,2)));
// CHECK: FunctionDecl{{.*}}TestBool
// CHECK: typeDetails: BuiltinType
// CHECK:   attrDetails: ArgumentWithTypeTagAttr{{.*}}pointer_with_type_tag bool1 1 2 IsPointer

void TestUnsigned(void *, int)
__attribute__((pointer_with_type_tag(unsigned1,1,2)));
// CHECK: FunctionDecl{{.*}}TestUnsigned
// CHECK: typeDetails: BuiltinType
// CHECK:   attrDetails: ArgumentWithTypeTagAttr{{.*}} pointer_with_type_tag unsigned1 1 2

void TestInt(void) __attribute__((constructor(123)));
// CHECK:      FunctionDecl{{.*}}TestInt
// CHECK-NEXT:   attrDetails: ConstructorAttr{{.*}} 123

static int TestString __attribute__((alias("alias1")));
// CHECK:      VarDecl{{.*}}TestString
// CHECK-NEXT: typeDetails: BuiltinType
// CHECK-NEXT:   attrDetails: AliasAttr{{.*}} "alias1"

extern struct s1 TestType
__attribute__((type_tag_for_datatype(ident1,int)));
// CHECK:      VarDecl{{.*}}TestType
// CHECK-NEXT: |-typeDetails: ElaboratedType 0x{{.+}} 'struct s1' sugar
// CHECK-NEXT: | `-typeDetails: RecordType 0x{{.+}} 's1'
// CHECK-NEXT: |   `-CXXRecord 0x{{.+}} 's1'
// CHECK-NEXT: `-attrDetails: TypeTagForDatatypeAttr 0x{{.+}} <{{.+}}> ident1 int

void TestLabel() {
L: __attribute__((unused)) int i;
// CHECK: LabelStmt{{.*}}'L'
// CHECK: VarDecl{{.*}}i 'int'
// CHECK-NEXT: typeDetails: BuiltinType
// CHECK-NEXT: attrDetails: UnusedAttr{{.*}}

M: __attribute(()) int j;
// CHECK: LabelStmt {{.*}} 'M'
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} j 'int'

N: __attribute(()) ;
// CHECK: LabelStmt {{.*}} 'N'
// CHECK-NEXT: NullStmt
}

namespace Test {
extern "C" int printf(const char *format, ...);
// CHECK: | `-FunctionDecl {{.*}} printf 'int (const char *, ...)'
// CHECK: |   |-ParmVarDecl {{.*}} format 'const char *'
// CHECK: |   | `-typeDetails: PointerType {{.*}} 'const char *'
// CHECK: |   |   `-qualTypeDetail: QualType {{.*}} 'const char' const
// CHECK: |   |     `-typeDetails: BuiltinType {{.*}} 'char'
// CHECK: |   |-attrDetails: BuiltinAttr {{.*}} <<invalid sloc>> Implicit 1059
// CHECK: |   `-attrDetails: FormatAttr {{.*}} Implicit printf 1 2

alignas(8) extern int x;
extern int x;
// CHECK: VarDecl{{.*}} x 'int'
// CHECK: VarDecl{{.*}} x 'int'
// CHECK-NEXT: typeDetails: BuiltinType
// CHECK-NEXT: attrDetails: AlignedAttr{{.*}} Inherited
}

namespace TestAligns {

template<typename...T> struct my_union {
  alignas(T...) char buffer[1024];
};

template<typename...T> struct my_union2 {
  _Alignas(T...) char buffer[1024];
};

struct alignas(8) A { char c; };
struct alignas(4) B { short s; };
struct C { char a[16]; };

// CHECK: ClassTemplateSpecializationDecl {{.*}} struct my_union

// CHECK: CXXRecordDecl {{.*}} implicit struct my_union
// CHECK: FieldDecl {{.*}} buffer 'char[1024]'
// CHECK-NEXT: attrDetails: AlignedAttr {{.*}} alignas 'TestAligns::A'
// CHECK-NEXT: attrDetails: AlignedAttr {{.*}} alignas 'TestAligns::B'
// CHECK-NEXT: attrDetails: AlignedAttr {{.*}} alignas 'TestAligns::C'
my_union<A, B, C> my_union_val;

// CHECK: ClassTemplateSpecializationDecl {{.*}} struct my_union2
// CHECK: CXXRecordDecl {{.*}} implicit struct my_union2
// CHECK: FieldDecl {{.*}} buffer 'char[1024]'
// CHECK-NEXT: attrDetails: AlignedAttr {{.*}} _Alignas 'TestAligns::A'
// CHECK-NEXT: attrDetails: AlignedAttr {{.*}} _Alignas 'TestAligns::B'
// CHECK-NEXT: attrDetails: AlignedAttr {{.*}} _Alignas 'TestAligns::C'
my_union2<A, B, C> my_union2_val;

} // namespace TestAligns

int __attribute__((cdecl)) TestOne(void), TestTwo(void);
// CHECK: FunctionDecl{{.*}}TestOne{{.*}}__attribute__((cdecl))
// CHECK: FunctionDecl{{.*}}TestTwo{{.*}}__attribute__((cdecl))

void func() {
  auto Test = []() __attribute__((no_thread_safety_analysis)) {};
  // CHECK: CXXMethodDecl{{.*}}operator() 'void () const'
  // CHECK: NoThreadSafetyAnalysisAttr

  // Because GNU's noreturn applies to the function type, and this lambda does
  // not have a capture list, the call operator and the function pointer
  // conversion should both be noreturn, but the method should not contain a
  // NoReturnAttr because the attribute applied to the type.
  auto Test2 = []() __attribute__((noreturn)) { while(1); };
  // CHECK: CXXMethodDecl{{.*}}operator() 'void () __attribute__((noreturn)) const'
  // CHECK-NOT: NoReturnAttr
  // CHECK: CXXConversionDecl{{.*}}operator void (*)() __attribute__((noreturn))
}

namespace PR20930 {
struct S {
  struct { int Test __attribute__((deprecated)); };
  // CHECK: FieldDecl{{.*}}Test 'int'
  // CHECK-NEXT: DeprecatedAttr
};

void f() {
  S s;
  s.Test = 1;
  // CHECK: IndirectFieldDecl{{.*}}Test 'int'
  // CHECK: attrDetails: DeprecatedAttr
}
}

struct __attribute__((objc_bridge_related(NSParagraphStyle,,))) TestBridgedRef;
// CHECK: CXXRecordDecl{{.*}} struct TestBridgedRef
// CHECK-NEXT: attrDetails: ObjCBridgeRelatedAttr{{.*}} NSParagraphStyle

void TestExternalSourceSymbolAttr1()
__attribute__((external_source_symbol(language="Swift", defined_in="module", generated_declaration)));
// CHECK: FunctionDecl{{.*}} TestExternalSourceSymbolAttr1
// CHECK-NEXT: attrDetails: ExternalSourceSymbolAttr{{.*}} "Swift" "module" GeneratedDeclaration

void TestExternalSourceSymbolAttr2()
__attribute__((external_source_symbol(defined_in="module", language="Swift")));
// CHECK: FunctionDecl{{.*}} TestExternalSourceSymbolAttr2
// CHECK-NEXT: attrDetails: ExternalSourceSymbolAttr{{.*}} "Swift" "module" ""{{$}}

void TestExternalSourceSymbolAttr3()
__attribute__((external_source_symbol(generated_declaration, language="Objective-C++", defined_in="module")));
// CHECK: FunctionDecl{{.*}} TestExternalSourceSymbolAttr3
// CHECK-NEXT: attrDetails: ExternalSourceSymbolAttr{{.*}} "Objective-C++" "module" GeneratedDeclaration

void TestExternalSourceSymbolAttr4()
__attribute__((external_source_symbol(defined_in="Some external file.cs", generated_declaration, language="C Sharp")));
// CHECK: FunctionDecl{{.*}} TestExternalSourceSymbolAttr4
// CHECK-NEXT: attrDetails: ExternalSourceSymbolAttr{{.*}} "C Sharp" "Some external file.cs" GeneratedDeclaration

void TestExternalSourceSymbolAttr5()
__attribute__((external_source_symbol(generated_declaration, defined_in="module", language="Swift")));
// CHECK: FunctionDecl{{.*}} TestExternalSourceSymbolAttr5
// CHECK-NEXT: attrDetails: ExternalSourceSymbolAttr{{.*}} "Swift" "module" GeneratedDeclaration

void TestExternalSourceSymbolAttr6()
__attribute__((external_source_symbol(generated_declaration, defined_in="module", language="Swift", USR="testUSR")));
// CHECK: FunctionDecl{{.*}} TestExternalSourceSymbolAttr6
// CHECK-NEXT: attrDetails: ExternalSourceSymbolAttr{{.*}} "Swift" "module" GeneratedDeclaration "testUSR"

namespace TestNoEscape {
  void noescapeFunc(int *p0, __attribute__((noescape)) int *p1) {}
  // CHECK: `-FunctionDecl{{.*}} noescapeFunc 'void (int *, __attribute__((noescape)) int *)'
  // CHECK: ParmVarDecl
  // CHECK: ParmVarDecl
  // CHECK: -attrDetails: NoEscapeAttr
}

namespace TestSuppress {
  [[gsl::suppress("at-namespace")]];
  // CHECK: NamespaceDecl{{.*}} TestSuppress
  // CHECK-NEXT: EmptyDecl{{.*}}
  // CHECK-NEXT: SuppressAttr{{.*}} at-namespace
  [[gsl::suppress("on-decl")]]
  void TestSuppressFunction();
  // CHECK: FunctionDecl{{.*}} TestSuppressFunction
  // CHECK-NEXT: SuppressAttr{{.*}} on-decl

  void f() {
      int *i;

      [[gsl::suppress("on-stmt")]] {
      // CHECK: AttributedStmt
      // CHECK-NEXT: SuppressAttr{{.*}} on-stmt
      // CHECK-NEXT: CompoundStmt
        i = reinterpret_cast<int*>(7);
      }
    }
}

namespace TestLifetimeCategories {
class [[gsl::Owner(int)]] AOwner{};
// CHECK: CXXRecordDecl{{.*}} class AOwner
// CHECK: attrDetails: OwnerAttr {{.*}} int
class [[gsl::Pointer(int)]] APointer{};
// CHECK: CXXRecordDecl{{.*}} class APointer
// CHECK: attrDetails: PointerAttr {{.*}} int

class [[gsl::Pointer]] PointerWithoutArgument{};
// CHECK: CXXRecordDecl{{.*}} class PointerWithoutArgument
// CHECK: attrDetails: PointerAttr

class [[gsl::Owner]] OwnerWithoutArgument{};
// CHECK: CXXRecordDecl{{.*}} class OwnerWithoutArgument
// CHECK: attrDetails: OwnerAttr
} // namespace TestLifetimeCategories

// Verify the order of attributes in the Ast. It must reflect the order
// in the parsed source.
int mergeAttrTest() __attribute__((deprecated)) __attribute__((warn_unused_result));
int mergeAttrTest() __attribute__((annotate("test")));
int mergeAttrTest() __attribute__((unused,no_thread_safety_analysis));
// CHECK: FunctionDecl{{.*}} mergeAttrTest
// CHECK-NEXT: attrDetails: DeprecatedAttr
// CHECK-NEXT: attrDetails: WarnUnusedResultAttr

// CHECK: FunctionDecl{{.*}} mergeAttrTest
// CHECK-NEXT: attrDetails: DeprecatedAttr{{.*}} Inherited
// CHECK-NEXT: attrDetails: WarnUnusedResultAttr{{.*}} Inherited
// CHECK-NEXT: attrDetails: AnnotateAttr{{.*}}

// CHECK: FunctionDecl{{.*}} mergeAttrTest
// CHECK-NEXT: attrDetails: DeprecatedAttr{{.*}} Inherited
// CHECK-NEXT: attrDetails: WarnUnusedResultAttr{{.*}} Inherited
// CHECK-NEXT: attrDetails: AnnotateAttr{{.*}} Inherited
// CHECK-NEXT: attrDetails: UnusedAttr
// CHECK-NEXT: attrDetails: NoThreadSafetyAnalysisAttr
