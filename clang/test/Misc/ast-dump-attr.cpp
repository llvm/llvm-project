// RUN: %clang_cc1 -triple x86_64-pc-linux -std=c++11 -Wno-deprecated-declarations -ast-dump -ast-dump-filter Test %s | FileCheck --strict-whitespace %s

int TestLocation
__attribute__((unused));
// CHECK:      VarDecl{{.*}}TestLocation
// CHECK-NEXT:   UnusedAttr 0x{{[^ ]*}} <line:[[@LINE-2]]:16>

int TestIndent
__attribute__((unused));
// CHECK:      {{^}}VarDecl{{.*TestIndent[^()]*$}}
// CHECK-NEXT: {{^}}`-UnusedAttr{{[^()]*$}}

void TestAttributedStmt() {
  switch (1) {
  case 1:
    [[clang::fallthrough]];
  case 2:
    ;
  }
}
// CHECK:      FunctionDecl{{.*}}TestAttributedStmt
// CHECK:      AttributedStmt
// CHECK-NEXT:   FallThroughAttr
// CHECK-NEXT:   NullStmt

[[clang::warn_unused_result]] int TestCXX11DeclAttr();
// CHECK:      FunctionDecl{{.*}}TestCXX11DeclAttr
// CHECK-NEXT:   WarnUnusedResultAttr

int TestAlignedNull __attribute__((aligned));
// CHECK:      VarDecl{{.*}}TestAlignedNull
// CHECK-NEXT:   AlignedAttr {{.*}} aligned
// CHECK-NEXT:     <<<NULL>>>

int TestAlignedExpr __attribute__((aligned(4)));
// CHECK:      VarDecl{{.*}}TestAlignedExpr
// CHECK-NEXT:   AlignedAttr {{.*}} aligned
// CHECK-NEXT:     IntegerLiteral

int TestEnum __attribute__((visibility("default")));
// CHECK:      VarDecl{{.*}}TestEnum
// CHECK-NEXT:   VisibilityAttr{{.*}} Default

class __attribute__((lockable)) Mutex {
} mu1, mu2;
int TestExpr __attribute__((guarded_by(mu1)));
// CHECK:      VarDecl{{.*}}TestExpr
// CHECK-NEXT:   GuardedByAttr
// CHECK-NEXT:     DeclRefExpr{{.*}}mu1

class Mutex TestVariadicExpr __attribute__((acquired_after(mu1, mu2)));
// CHECK:      VarDecl{{.*}}TestVariadicExpr
// CHECK:        AcquiredAfterAttr
// CHECK-NEXT:     DeclRefExpr{{.*}}mu1
// CHECK-NEXT:     DeclRefExpr{{.*}}mu2

void function1(void *) {
  int TestFunction __attribute__((cleanup(function1)));
}
// CHECK:      VarDecl{{.*}}TestFunction
// CHECK-NEXT:   CleanupAttr{{.*}} Function{{.*}}function1

void TestIdentifier(void *, int)
__attribute__((pointer_with_type_tag(ident1,1,2)));
// CHECK: FunctionDecl{{.*}}TestIdentifier
// CHECK:   ArgumentWithTypeTagAttr{{.*}} pointer_with_type_tag ident1

void TestBool(void *, int)
__attribute__((pointer_with_type_tag(bool1,1,2)));
// CHECK: FunctionDecl{{.*}}TestBool
// CHECK:   ArgumentWithTypeTagAttr{{.*}}pointer_with_type_tag bool1 0 1 IsPointer

void TestUnsigned(void *, int)
__attribute__((pointer_with_type_tag(unsigned1,1,2)));
// CHECK: FunctionDecl{{.*}}TestUnsigned
// CHECK:   ArgumentWithTypeTagAttr{{.*}} pointer_with_type_tag unsigned1 0 1

void TestInt(void) __attribute__((constructor(123)));
// CHECK:      FunctionDecl{{.*}}TestInt
// CHECK-NEXT:   ConstructorAttr{{.*}} 123

static int TestString __attribute__((alias("alias1")));
// CHECK:      VarDecl{{.*}}TestString
// CHECK-NEXT:   AliasAttr{{.*}} "alias1"

extern struct s1 TestType
__attribute__((type_tag_for_datatype(ident1,int)));
// CHECK:      VarDecl{{.*}}TestType
// CHECK-NEXT:   TypeTagForDatatypeAttr{{.*}} int

void TestLabel() {
L: __attribute__((unused)) int i;
// CHECK: LabelStmt{{.*}}'L'
// CHECK: VarDecl{{.*}}i 'int'
// CHECK-NEXT: UnusedAttr{{.*}}

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
// CHECK: FunctionDecl{{.*}}printf
// CHECK-NEXT: ParmVarDecl{{.*}}format{{.*}}'const char *'
// CHECK-NEXT: FormatAttr{{.*}}Implicit printf 1 2

alignas(8) extern int x;
extern int x;
// CHECK: VarDecl{{.*}} x 'int'
// CHECK: VarDecl{{.*}} x 'int'
// CHECK-NEXT: AlignedAttr{{.*}} Inherited
}

int __attribute__((cdecl)) TestOne(void), TestTwo(void);
// CHECK: FunctionDecl{{.*}}TestOne{{.*}}__attribute__((cdecl))
// CHECK: FunctionDecl{{.*}}TestTwo{{.*}}__attribute__((cdecl))

void func() {
  auto Test = []() __attribute__((no_thread_safety_analysis)) {};
  // CHECK: CXXMethodDecl{{.*}}operator() 'void (void) const'
  // CHECK: NoThreadSafetyAnalysisAttr

  // Because GNU's noreturn applies to the function type, and this lambda does
  // not have a capture list, the call operator and the function pointer
  // conversion should both be noreturn, but the method should not contain a
  // NoReturnAttr because the attribute applied to the type.
  auto Test2 = []() __attribute__((noreturn)) { while(1); };
  // CHECK: CXXMethodDecl{{.*}}operator() 'void (void) __attribute__((noreturn)) const'
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
  // CHECK: DeprecatedAttr
}
}

struct __attribute__((objc_bridge_related(NSParagraphStyle,,))) TestBridgedRef;
// CHECK: CXXRecordDecl{{.*}} struct TestBridgedRef
// CHECK-NEXT: ObjCBridgeRelatedAttr{{.*}} NSParagraphStyle

void TestExternalSourceSymbolAttr1()
__attribute__((external_source_symbol(language="Swift", defined_in="module", generated_declaration)));
// CHECK: FunctionDecl{{.*}} TestExternalSourceSymbolAttr1
// CHECK-NEXT: ExternalSourceSymbolAttr{{.*}} "Swift" "module" GeneratedDeclaration

void TestExternalSourceSymbolAttr2()
__attribute__((external_source_symbol(defined_in="module", language="Swift")));
// CHECK: FunctionDecl{{.*}} TestExternalSourceSymbolAttr2
// CHECK-NEXT: ExternalSourceSymbolAttr{{.*}} "Swift" "module"{{$}}

void TestExternalSourceSymbolAttr3()
__attribute__((external_source_symbol(generated_declaration, language="Objective-C++", defined_in="module")));
// CHECK: FunctionDecl{{.*}} TestExternalSourceSymbolAttr3
// CHECK-NEXT: ExternalSourceSymbolAttr{{.*}} "Objective-C++" "module" GeneratedDeclaration

void TestExternalSourceSymbolAttr4()
__attribute__((external_source_symbol(defined_in="Some external file.cs", generated_declaration, language="C Sharp")));
// CHECK: FunctionDecl{{.*}} TestExternalSourceSymbolAttr4
// CHECK-NEXT: ExternalSourceSymbolAttr{{.*}} "C Sharp" "Some external file.cs" GeneratedDeclaration

void TestExternalSourceSymbolAttr5()
__attribute__((external_source_symbol(generated_declaration, defined_in="module", language="Swift")));
// CHECK: FunctionDecl{{.*}} TestExternalSourceSymbolAttr5
// CHECK-NEXT: ExternalSourceSymbolAttr{{.*}} "Swift" "module" GeneratedDeclaration
