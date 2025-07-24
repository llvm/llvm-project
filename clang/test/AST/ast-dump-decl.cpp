// Test without serialization:
// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -fms-extensions \
// RUN: -ast-dump -ast-dump-filter Test %s \
// RUN: | FileCheck --strict-whitespace %s
//
// Test with serialization: FIXME: Find why the outputs differs and fix it!
//    : %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -fms-extensions -emit-pch -o %t %s
//    : %clang_cc1 -x c++ -std=c++11 -triple x86_64-linux-gnu -fms-extensions -include-pch %t \
//    : -ast-dump-all -ast-dump-filter Test /dev/null \
//    : | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
//    : | FileCheck --strict-whitespace %s

class testEnumDecl {
  enum class TestEnumDeclScoped;
  enum TestEnumDeclFixed : int;
};
// CHECK: EnumDecl{{.*}} class TestEnumDeclScoped 'int'
// CHECK-NOT: instantiated_from
// CHECK: EnumDecl{{.*}} TestEnumDeclFixed 'int'
// CHECK-NOT: instantiated_from

class testFieldDecl {
  int TestFieldDeclInit = 0;
};
// CHECK:      FieldDecl{{.*}} TestFieldDeclInit 'int'
// CHECK:        IntegerLiteral

namespace testVarDeclNRVO {
  class A { };
  A TestFuncNRVO() {
    A TestVarDeclNRVO;
    return TestVarDeclNRVO;
  }
}
// CHECK:      FunctionDecl 0x{{.+}} TestFuncNRVO 'A ()'
// CHECK-NEXT: `-CompoundStmt 0x{{.+}} 
// CHECK-NEXT:   |-DeclStmt 0x{{.+}} 
// CHECK-NEXT:   | `-VarDecl 0x{{.+}} used TestVarDeclNRVO 'A':'testVarDeclNRVO::A' nrvo callinit
// CHECK-NEXT:   |   |-CXXConstructExpr 0x{{.+}} 'A':'testVarDeclNRVO::A' 'void () noexcept'
// CHECK-NEXT:   |   `-typeDetails: ElaboratedType 0x{{.+}} 'A' sugar
// CHECK-NEXT:   |     `-typeDetails: RecordType 0x{{.+}} 'testVarDeclNRVO::A'
// CHECK-NEXT:   |       `-CXXRecord 0x{{.+}} 'A'
// CHECK-NEXT:   `-ReturnStmt 0x{{.+}} nrvo_candidate(Var 0x{{.+}} 'TestVarDeclNRVO' 'A':'testVarDeclNRVO::A')
// CHECK-NEXT:     `-CXXConstructExpr 0x{{.+}} 'A':'testVarDeclNRVO::A' 'void (A &&) noexcept'
// CHECK-NEXT:       `-ImplicitCastExpr 0x{{.+}} 'A':'testVarDeclNRVO::A' xvalue <NoOp>
// CHECK-NEXT:         `-DeclRefExpr 0x{{.+}} 'A':'testVarDeclNRVO::A' lvalue Var 0x{{.+}} 'TestVarDeclNRVO' 'A':'testVarDeclNRVO::A'

void testParmVarDeclInit(int TestParmVarDeclInit = 0);
// CHECK:      ParmVarDecl{{.*}} TestParmVarDeclInit 'int'
// CHECK:        IntegerLiteral{{.*}}

namespace TestNamespaceDecl {
  int i;
}
// CHECK:      NamespaceDecl{{.*}} TestNamespaceDecl
// CHECK:        VarDecl

namespace TestNamespaceDecl {
  int j;
}
// CHECK:      NamespaceDecl{{.*}} TestNamespaceDecl
// CHECK:        original Namespace
// CHECK:        VarDecl

inline namespace TestNamespaceDeclInline {
}
// CHECK:      NamespaceDecl{{.*}} TestNamespaceDeclInline inline

namespace TestNestedNameSpace::Nested {
}
// CHECK:      NamespaceDecl{{.*}} TestNestedNameSpace
// CHECK:      NamespaceDecl{{.*}} Nested nested{{\s*$}}

namespace TestMultipleNested::SecondLevelNested::Nested {
}
// CHECK:      NamespaceDecl{{.*}} TestMultipleNested
// CHECK:      NamespaceDecl{{.*}} SecondLevelNested nested
// CHECK:      NamespaceDecl{{.*}} Nested nested{{\s*$}}

namespace TestInlineNested::inline SecondLevel::inline Nested {
}
// CHECK:      NamespaceDecl{{.*}} TestInlineNested
// CHECK:      NamespaceDecl{{.*}} SecondLevel inline nested
// CHECK:      NamespaceDecl{{.*}} Nested inline nested{{\s*$}}

namespace testUsingDirectiveDecl {
  namespace A {
  }
}
namespace TestUsingDirectiveDecl {
  using namespace testUsingDirectiveDecl::A;
}
// CHECK:      NamespaceDecl{{.*}} TestUsingDirectiveDecl
// CHECK:        UsingDirectiveDecl{{.*}} Namespace{{.*}} 'A'

namespace testNamespaceAlias {
  namespace A {
  }
}
namespace TestNamespaceAlias = testNamespaceAlias::A;
// CHECK:      NamespaceAliasDecl{{.*}} TestNamespaceAlias
// CHECK:        Namespace{{.*}} 'A'

using TestTypeAliasDecl = int;
// CHECK: TypeAliasDecl{{.*}} TestTypeAliasDecl 'int'

namespace testTypeAliasTemplateDecl {
  template<typename T> class A;
  template<typename T> using TestTypeAliasTemplateDecl = A<T>;
}
// CHECK:      TypeAliasTemplateDecl{{.*}} TestTypeAliasTemplateDecl
// CHECK:        TemplateTypeParmDecl
// CHECK:        TypeAliasDecl{{.*}} TestTypeAliasTemplateDecl 'A<T>'

namespace testCXXRecordDecl {
  class TestEmpty {};
// CHECK:      CXXRecordDecl{{.*}} class TestEmpty
// CHECK:        DefinitionData pass_in_registers empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK:          DefaultConstructor exists trivial constexpr
// CHECK:          CopyConstructor simple trivial has_const_param
// CHECK:          MoveConstructor exists simple trivial
// CHECK:          CopyAssignment simple trivial has_const_param
// CHECK:          MoveAssignment exists simple trivial
// CHECK:          Destructor simple irrelevant trivial

  class A { };
  class B { };
  class TestCXXRecordDecl : virtual A, public B {
    int i;
  };
}
// CHECK:      CXXRecordDecl{{.*}} class TestCXXRecordDecl
// CHECK:        DefinitionData{{$}}
// CHECK:          DefaultConstructor exists non_trivial
// CHECK:          CopyConstructor simple non_trivial has_const_param
// CHECK:          MoveConstructor exists simple non_trivial
// CHECK:          CopyAssignment simple non_trivial has_const_param
// CHECK:          MoveAssignment exists simple non_trivial
// CHECK:          Destructor simple irrelevant trivial
// CHECK:        virtual private 'A':'testCXXRecordDecl::A'
// CHECK:        public 'B':'testCXXRecordDecl::B'
// CHECK:        CXXRecordDecl{{.*}} class TestCXXRecordDecl
// CHECK:        FieldDecl

template<class...T>
class TestCXXRecordDeclPack : public T... {
};
// CHECK:      CXXRecordDecl{{.*}} class TestCXXRecordDeclPack
// CHECK:        public 'T'...
// CHECK:        CXXRecordDecl{{.*}} class TestCXXRecordDeclPack

thread_local int TestThreadLocalInt;
// CHECK: TestThreadLocalInt {{.*}} tls_dynamic

class testCXXMethodDecl {
  virtual void TestCXXMethodDeclPure() = 0;
  void TestCXXMethodDeclDelete() = delete;
  void TestCXXMethodDeclThrow() throw();
  void TestCXXMethodDeclThrowType() throw(int);
};
// CHECK: CXXMethodDecl{{.*}} TestCXXMethodDeclPure 'void ()' virtual pure
// CHECK: CXXMethodDecl{{.*}} TestCXXMethodDeclDelete 'void ()' delete
// CHECK: CXXMethodDecl{{.*}} TestCXXMethodDeclThrow 'void () throw()'
// CHECK: CXXMethodDecl{{.*}} TestCXXMethodDeclThrowType 'void () throw(int)'

namespace testCXXConstructorDecl {
  class A { };
  class TestCXXConstructorDecl : public A {
    int I;
    TestCXXConstructorDecl(A &a, int i) : A(a), I(i) { }
    TestCXXConstructorDecl(A &a) : TestCXXConstructorDecl(a, 0) { }
  };
}
// CHECK:      CXXConstructorDecl{{.*}} TestCXXConstructorDecl 'void {{.*}}'
// CHECK:        ParmVarDecl{{.*}} a
// CHECK:        ParmVarDecl{{.*}} i
// CHECK:        CXXCtorInitializer{{.*}}A
// CHECK:          Expr
// CHECK:        CXXCtorInitializer{{.*}}I
// CHECK:          Expr
// CHECK:        CompoundStmt
// CHECK:      CXXConstructorDecl{{.*}} TestCXXConstructorDecl 'void {{.*}}'
// CHECK:        ParmVarDecl{{.*}} a
// CHECK:        CXXCtorInitializer{{.*}}TestCXXConstructorDecl
// CHECK:          CXXConstructExpr{{.*}}TestCXXConstructorDecl

class TestCXXDestructorDecl {
  ~TestCXXDestructorDecl() { }
};
// CHECK:      CXXDestructorDecl{{.*}} ~TestCXXDestructorDecl 'void () noexcept'
// CHECK:        CompoundStmt

// Test that the range of a defaulted members is computed correctly.
class TestMemberRanges {
public:
  TestMemberRanges() = default;
  TestMemberRanges(const TestMemberRanges &Other) = default;
  TestMemberRanges(TestMemberRanges &&Other) = default;
  ~TestMemberRanges() = default;
  TestMemberRanges &operator=(const TestMemberRanges &Other) = default;
  TestMemberRanges &operator=(TestMemberRanges &&Other) = default;
};
void SomeFunction() {
  TestMemberRanges A;
  TestMemberRanges B(A);
  B = A;
  A = static_cast<TestMemberRanges &&>(B);
  TestMemberRanges C(static_cast<TestMemberRanges &&>(A));
}
// CHECK:      CXXConstructorDecl{{.*}} <line:{{.*}}:3, col:30>
// CHECK:      CXXConstructorDecl{{.*}} <line:{{.*}}:3, col:59>
// CHECK:      CXXConstructorDecl{{.*}} <line:{{.*}}:3, col:54>
// CHECK:      CXXDestructorDecl{{.*}} <line:{{.*}}:3, col:31>
// CHECK:      CXXMethodDecl{{.*}} <line:{{.*}}:3, col:70>
// CHECK:      CXXMethodDecl{{.*}} <line:{{.*}}:3, col:65>

class TestCXXConversionDecl {
  operator int() { return 0; }
};
// CHECK:      CXXConversionDecl{{.*}} operator int 'int ()'
// CHECK:        CompoundStmt

namespace TestStaticAssertDecl {
  static_assert(true, "msg");
}
// CHECK:      NamespaceDecl{{.*}} TestStaticAssertDecl
// CHECK:        StaticAssertDecl{{.*> .*$}}
// CHECK:          CXXBoolLiteralExpr
// CHECK:          StringLiteral

namespace testFunctionTemplateDecl {
  class A { };
  class B { };
  class C { };
  class D { };
  template<typename T> void TestFunctionTemplate(T) { }

  // implicit instantiation
  void bar(A a) { TestFunctionTemplate(a); }

  // explicit specialization
  template<> void TestFunctionTemplate(B);

  // explicit instantiation declaration
  extern template void TestFunctionTemplate(C);

  // explicit instantiation definition
  template void TestFunctionTemplate(D);
}
  // CHECK:  FunctionTemplateDecl 0x{{.+}} <{{.+}}:[[@LINE-14]]:3, col:55> col:29 TestFunctionTemplate
  // CHECK:  |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 referenced typename depth 0 index 0 T
  // CHECK:  |-FunctionDecl 0x{{.+}} <col:24, col:55> col:29 TestFunctionTemplate 'void (T)'
  // CHECK:  | |-ParmVarDecl 0x{{.+}} <col:50> col:51 'T'
  // CHECK:  | `-CompoundStmt 0x{{.+}} <col:53, col:55>
  // CHECK:  |-FunctionDecl 0x{{.+}} <col:24, col:55> col:29 used TestFunctionTemplate 'void (testFunctionTemplateDecl::A)'
  // CHECK:  | |-TemplateArgument type 'testFunctionTemplateDecl::A'
  // CHECK:  | | `-typeDetails: RecordType 0{{.+}} 'testFunctionTemplateDecl::A'
  // CHECK:  | |   `-CXXRecord 0x{{.+}} 'A'
  // CHECK:  | |-ParmVarDecl 0x{{.+}} <col:50> col:51 'testFunctionTemplateDecl::A'
  // CHECK:  | `-CompoundStmt 0x{{.+}} <col:53, col:55>
  // CHECK:  |-Function 0x{{.+}} 'TestFunctionTemplate' 'void (B)'
  // CHECK:  |-FunctionDecl 0x{{.+}} <col:24, col:55> col:29 TestFunctionTemplate 'void (testFunctionTemplateDecl::C)'
  // CHECK:  | |-TemplateArgument type 'testFunctionTemplateDecl::C'
  // CHECK:  | | `-typeDetails: RecordType 0{{.+}} 'testFunctionTemplateDecl::C'
  // CHECK:  | |   `-CXXRecord 0x{{.+}} 'C'
  // CHECK:  | `-ParmVarDecl 0x{{.+}} <col:50> col:51 'testFunctionTemplateDecl::C'
  // CHECK:  `-FunctionDecl 0x{{.+}} <col:24, col:55> col:29 TestFunctionTemplate 'void (testFunctionTemplateDecl::D)'
  // CHECK:    |-TemplateArgument type 'testFunctionTemplateDecl::D'
  // CHECK:    | `-typeDetails: RecordType 0{{.+}} 'testFunctionTemplateDecl::D'
  // CHECK:    |   `-CXXRecord 0x{{.+}} 'D'
  // CHECK:    |-ParmVarDecl 0x{{.+}} <col:50> col:51 'testFunctionTemplateDecl::D'
  // CHECK:    `-CompoundStmt 0x{{.+}} <col:53, col:55>

  // CHECK:  FunctionDecl 0x{{.+}} prev 0x{{.+}} <{{.+}}:[[@LINE-32]]:3, col:41> col:19 TestFunctionTemplate 'void (B)'
  // CHECK:  |-TemplateArgument type 'testFunctionTemplateDecl::B'
  // CHECK:  | `-typeDetails: RecordType 0{{.+}} 'testFunctionTemplateDecl::B'
  // CHECK:  |   `-CXXRecord 0x{{.+}} 'B'
  // CHECK:  `-ParmVarDecl 0x{{.+}} <col:40> col:41 'B':'testFunctionTemplateDecl::B'


namespace testClassTemplateDecl {
  class A { };
  class B { };
  class C { };
  class D { };

  template<typename T> class TestClassTemplate {
  public:
    TestClassTemplate();
    ~TestClassTemplate();
    int j();
    int i;
  };

  // implicit instantiation
  TestClassTemplate<A> a;

  // explicit specialization
  template<> class TestClassTemplate<B> {
    int j;
  };

  // explicit instantiation declaration
  extern template class TestClassTemplate<C>;

  // explicit instantiation definition
  template class TestClassTemplate<D>;

  // partial explicit specialization
  template<typename T1, typename T2> class TestClassTemplatePartial {
    int i;
  };
  template<typename T1> class TestClassTemplatePartial<T1, A> {
    int j;
  };

  template<typename T = int> struct TestTemplateDefaultType;
  template<typename T> struct TestTemplateDefaultType { };

  template<int I = 42> struct TestTemplateDefaultNonType;
  template<int I> struct TestTemplateDefaultNonType { };

  template<template<typename> class TT = TestClassTemplate> struct TestTemplateTemplateDefaultType;
  template<template<typename> class TT> struct TestTemplateTemplateDefaultType { };
}

// CHECK:       ClassTemplateDecl 0x{{.+}} <{{.+}}:[[@LINE-40]]:3, line:[[@LINE-34]]:3> line:[[@LINE-40]]:30 TestClassTemplate{{$}}
// CHECK:       |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 typename depth 0 index 0 T{{$}}
// CHECK:       |-CXXRecordDecl 0x{{.+}} <col:24, line:[[@LINE-36]]:3> line:[[@LINE-42]]:30 class TestClassTemplate definition{{$}}
// CHECK:       | |-DefinitionData standard_layout has_user_declared_ctor can_const_default_init{{$}}
// CHECK:       | | |-DefaultConstructor exists non_trivial user_provided{{$}}
// CHECK:       | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
// CHECK:       | | |-MoveConstructor{{$}}
// CHECK:       | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
// CHECK:       | | |-MoveAssignment{{$}}
// CHECK:       | | `-Destructor irrelevant non_trivial user_declared{{$}}
// CHECK:       | |-CXXRecordDecl 0x{{.+}} <col:24, col:30> col:30 implicit referenced class TestClassTemplate{{$}}
// CHECK:       | |-AccessSpecDecl 0x{{.+}} <line:[[@LINE-50]]:3, col:9> col:3 public{{$}}
// CHECK:       | |-CXXConstructorDecl 0x{{.+}} <line:[[@LINE-50]]:5, col:23> col:5 TestClassTemplate<T> 'void ()'{{$}}
// CHECK:       | |-CXXDestructorDecl 0x{{.+}} <line:[[@LINE-50]]:5, col:24> col:5 ~TestClassTemplate<T> 'void ()' not_selected{{$}}
// CHECK:       | |-CXXMethodDecl 0x{{.+}} <line:[[@LINE-50]]:5, col:11> col:9 j 'int ()'{{$}}
// CHECK:       | `-FieldDecl 0x{{.+}} <line:[[@LINE-50]]:5, col:9> col:9 i 'int'{{$}}
// CHECK:       |-ClassTemplateSpecializationDecl 0x{{.+}} <line:[[@LINE-56]]:3, line:[[@LINE-50]]:3> line:[[@LINE-56]]:30 class TestClassTemplate definition implicit_instantiation{{$}}
// CHECK:       | |-DefinitionData standard_layout has_user_declared_ctor can_const_default_init{{$}}
// CHECK:       | | |-DefaultConstructor exists non_trivial user_provided{{$}}
// CHECK:       | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param{{$}}
// CHECK:       | | |-MoveConstructor{{$}}
// CHECK:       | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
// CHECK:       | | |-MoveAssignment{{$}}
// CHECK:       | | `-Destructor non_trivial user_declared{{$}}
// CHECK:       | |-TemplateArgument type 'testClassTemplateDecl::A'{{$}}
// CHECK:       | | `-typeDetails: RecordType 0{{.+}} 'testClassTemplateDecl::A'{{$}}
// CHECK:       | |   `-CXXRecord 0x{{.+}} 'A'{{$}}
// CHECK:       | |-CXXRecordDecl 0x{{.+}} <col:24, col:30> col:30 implicit class TestClassTemplate{{$}}
// CHECK:       | |-AccessSpecDecl 0x{{.+}} <line:[[@LINE-67]]:3, col:9> col:3 public{{$}}
// CHECK:       | |-CXXConstructorDecl 0x{{.+}} <line:[[@LINE-67]]:5, col:23> col:5 used TestClassTemplate 'void ()' implicit_instantiation instantiated_from {{0x[^ ]+}}{{$}}
// CHECK:       | |-CXXDestructorDecl 0x{{.+}} <line:[[@LINE-67]]:5, col:24> col:5 used ~TestClassTemplate 'void () noexcept' implicit_instantiation instantiated_from {{0x[^ ]+}}{{$}}
// CHECK:       | |-CXXMethodDecl 0x{{.+}} <line:[[@LINE-67]]:5, col:11> col:9 j 'int ()' implicit_instantiation instantiated_from {{0x[^ ]+}}{{$}}
// CHECK:       | |-FieldDecl 0x{{.+}} <line:[[@LINE-67]]:5, col:9> col:9 i 'int'{{$}}
// CHECK:       | `-CXXConstructorDecl {{.+}} implicit constexpr TestClassTemplate 'void (const TestClassTemplate<testClassTemplateDecl::A> &)' inline default trivial noexcept-unevaluated
// CHECK:       |   `-ParmVarDecl 0x{{.+}} 'const TestClassTemplate<testClassTemplateDecl::A> &'
// CHECK:       |-ClassTemplateSpecialization 0x{{.+}} 'TestClassTemplate'{{$}}
// CHECK:       |-ClassTemplateSpecialization 0x{{.+}} 'TestClassTemplate'{{$}}
// CHECK:       `-ClassTemplateSpecialization 0x{{.+}} 'TestClassTemplate'{{$}}

// CHECK:       ClassTemplateSpecializationDecl 0x{{.+}} <{{.+}}:[[@LINE-67]]:3, line:[[@LINE-65]]:3> line:[[@LINE-67]]:20 class TestClassTemplate definition explicit_specialization{{$}}
// CHECK:       |-DefinitionData pass_in_registers standard_layout trivially_copyable trivial literal{{$}}
// CHECK:       | |-DefaultConstructor exists trivial needs_implicit{{$}}
// CHECK:       | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
// CHECK:       | |-MoveConstructor exists simple trivial needs_implicit{{$}}
// CHECK:       | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
// CHECK:       | |-MoveAssignment exists simple trivial needs_implicit{{$}}
// CHECK:       | `-Destructor simple irrelevant trivial needs_implicit{{$}}
// CHECK:       |-TemplateArgument type 'testClassTemplateDecl::B'{{$}}
// CHECK:       | `-typeDetails: RecordType 0{{.+}} 'testClassTemplateDecl::B'{{$}}
// CHECK:       |   `-CXXRecord 0x{{.+}} 'B'{{$}}
// CHECK:       |-CXXRecordDecl 0x{{.+}} <col:14, col:20> col:20 implicit class TestClassTemplate{{$}}
// CHECK:       `-FieldDecl 0x{{.+}} <line:[[@LINE-78]]:5, col:9> col:9 j 'int'{{$}}

// CHECK:       ClassTemplateSpecializationDecl 0x{{.+}} <{{.+}}:{{.*}}:3, col:44> col:25 class TestClassTemplate definition explicit_instantiation_declaration{{$}}
// CHECK:       |-DefinitionData standard_layout has_user_declared_ctor can_const_default_init{{$}}
// CHECK:       | |-DefaultConstructor exists non_trivial user_provided{{$}}
// CHECK:       | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
// CHECK:       | |-MoveConstructor{{$}}
// CHECK:       | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
// CHECK:       | |-MoveAssignment{{$}}
// CHECK:       | `-Destructor non_trivial user_declared{{$}}
// CHECK:       |-TemplateArgument type 'testClassTemplateDecl::C'{{$}}
// CHECK:       | `-typeDetails: RecordType 0{{.+}} 'testClassTemplateDecl::C'{{$}}
// CHECK:       |   `-CXXRecord 0x{{.+}} 'C'{{$}}
// CHECK:       |-CXXRecordDecl 0x{{.+}} <line:[[@LINE-104]]:24, col:30> col:30 implicit class TestClassTemplate{{$}}
// CHECK:       |-AccessSpecDecl 0x{{.+}} <line:[[@LINE-104]]:3, col:9> col:3 public{{$}}
// CHECK:       |-CXXConstructorDecl 0x{{.+}} <line:[[@LINE-104]]:5, col:23> col:5 TestClassTemplate 'void ()' explicit_instantiation_declaration instantiated_from {{0x[^ ]+}}{{$}}
// CHECK:       |-CXXDestructorDecl 0x{{.+}} <line:[[@LINE-104]]:5, col:24> col:5 ~TestClassTemplate 'void ()' explicit_instantiation_declaration noexcept-unevaluated 0x{{[^ ]+}} instantiated_from {{0x[^ ]+}}
// CHECK:       |-CXXMethodDecl 0x{{.+}} <line:[[@LINE-104]]:5, col:11> col:9 j 'int ()' explicit_instantiation_declaration instantiated_from {{0x[^ ]+}}{{$}}
// CHECK:       `-FieldDecl 0x{{.+}} <line:[[@LINE-104]]:5, col:9> col:9 i 'int'{{$}}

// CHECK:       ClassTemplateSpecializationDecl 0x{{.+}} <{{.+}}:[[@LINE-91]]:3, col:37> col:18 class TestClassTemplate definition explicit_instantiation_definition{{$}}
// CHECK:       |-DefinitionData standard_layout has_user_declared_ctor can_const_default_init{{$}}
// CHECK:       | |-DefaultConstructor exists non_trivial user_provided{{$}}
// CHECK:       | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
// CHECK:       | |-MoveConstructor{{$}}
// CHECK:       | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
// CHECK:       | |-MoveAssignment{{$}}
// CHECK:       | `-Destructor non_trivial user_declared{{$}}
// CHECK:       |-TemplateArgument type 'testClassTemplateDecl::D'{{$}}
// CHECK:       | `-typeDetails: RecordType 0{{.+}} 'testClassTemplateDecl::D'{{$}}
// CHECK:       |   `-CXXRecord 0x{{.+}} 'D'{{$}}
// CHECK:       |-CXXRecordDecl 0x{{.+}} <line:[[@LINE-122]]:24, col:30> col:30 implicit class TestClassTemplate{{$}}
// CHECK:       |-AccessSpecDecl 0x{{.+}} <line:[[@LINE-122]]:3, col:9> col:3 public{{$}}
// CHECK:       |-CXXConstructorDecl 0x{{.+}} <line:[[@LINE-122]]:5, col:23> col:5 TestClassTemplate 'void ()' implicit_instantiation instantiated_from {{0x[^ ]+}}{{$}}
// CHECK:       |-CXXDestructorDecl 0x{{.+}} <line:[[@LINE-122]]:5, col:24> col:5 ~TestClassTemplate 'void ()' implicit_instantiation noexcept-unevaluated 0x{{.+}} instantiated_from {{0x[^ ]+}}{{$}}
// CHECK:       |-CXXMethodDecl 0x{{.+}} <line:[[@LINE-122]]:5, col:11> col:9 j 'int ()' implicit_instantiation instantiated_from {{0x[^ ]+}}{{$}}
// CHECK:       `-FieldDecl 0x{{.+}} <line:[[@LINE-122]]:5, col:9> col:9 i 'int'{{$}}

// CHECK:      ClassTemplateDecl 0x{{.+}} <{{.+}}:[[@LINE-106]]:3, line:[[@LINE-104]]:3> line:[[@LINE-106]]:44 TestClassTemplatePartial{{$}}
// CHECK:       |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 typename depth 0 index 0 T1{{$}}
// CHECK:       |-TemplateTypeParmDecl 0x{{.+}} <col:25, col:34> col:34 typename depth 0 index 1 T2{{$}}
// CHECK:       `-CXXRecordDecl 0x{{.+}} <col:38, line:[[@LINE-107]]:3> line:[[@LINE-109]]:44 class TestClassTemplatePartial definition{{$}}
// CHECK:         |-DefinitionData standard_layout trivially_copyable trivial literal{{$}}
// CHECK:         | |-DefaultConstructor exists trivial needs_implicit{{$}}
// CHECK:         | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
// CHECK:         | |-MoveConstructor exists simple trivial needs_implicit{{$}}
// CHECK:         | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
// CHECK:         | |-MoveAssignment exists simple trivial needs_implicit{{$}}
// CHECK:         | `-Destructor simple irrelevant trivial needs_implicit{{$}}
// CHECK:         |-CXXRecordDecl 0x{{.+}} <col:38, col:44> col:44 implicit class TestClassTemplatePartial{{$}}
// CHECK:         `-FieldDecl 0x{{.+}} <line:[[@LINE-117]]:5, col:9> col:9 i 'int'{{$}}

// CHECK:       ClassTemplatePartialSpecializationDecl 0x{{.+}} <{{.+}}:[[@LINE-117]]:3, line:[[@LINE-115]]:3> line:[[@LINE-117]]:31 class TestClassTemplatePartial definition explicit_specialization{{$}}
// CHECK:       |-DefinitionData standard_layout trivially_copyable trivial literal{{$}}
// CHECK:       | |-DefaultConstructor exists trivial needs_implicit{{$}}
// CHECK:       | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
// CHECK:       | |-MoveConstructor exists simple trivial needs_implicit{{$}}
// CHECK:       | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
// CHECK:       | |-MoveAssignment exists simple trivial needs_implicit{{$}}
// CHECK:       | `-Destructor simple irrelevant trivial needs_implicit{{$}}
// CHECK:       |-TemplateArgument type 'type-parameter-0-0'{{$}}
// CHECK:       | `-typeDetails: TemplateTypeParmType 0x{{.+}} 'type-parameter-0-0' dependent depth 0 index 0{{$}}
// CHECK:       |-TemplateArgument type 'testClassTemplateDecl::A'{{$}}
// CHECK:       | `-typeDetails: RecordType 0x{{.+}} 'testClassTemplateDecl::A'{{$}}
// CHECK:       |   `-CXXRecord 0x{{.+}} 'A'{{$}}
// CHECK:       |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 referenced typename depth 0 index 0 T1{{$}}
// CHECK:       |-CXXRecordDecl 0x{{.+}} <col:25, col:31> col:31 implicit class TestClassTemplatePartial{{$}}
// CHECK:       `-FieldDecl 0x{{.+}} <line:[[@LINE-131]]:5, col:9> col:9 j 'int'{{$}}

// CHECK:       ClassTemplateDecl 0x{{.+}} <{{.+}}:[[@LINE-130]]:3, col:37> col:37 TestTemplateDefaultType{{$}}
// CHECK:       |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:25> col:21 typename depth 0 index 0 T{{$}}
// CHECK:       | `-TemplateArgument type 'int'{{$}}
// CHECK:       |   `-typeDetails: BuiltinType 0x{{.+}} 'int'{{$}}
// CHECK:       `-CXXRecordDecl 0x{{.+}} <col:30, col:37> col:37 struct TestTemplateDefaultType{{$}}

// CHECK:       ClassTemplateDecl 0x{{.+}} prev 0x{{.+}} <{{.+}}:[[@LINE-135]]:3, col:57> col:31 TestTemplateDefaultType{{$}}
// CHECK:       |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 typename depth 0 index 0 T{{$}}
// CHECK:       | `-TemplateArgument type 'int'{{$}}
// CHECK:       |   |-inherited from TemplateTypeParm 0x{{.+}} 'T'{{$}}
// CHECK:       |   `-typeDetails: BuiltinType 0x{{.+}} 'int'{{$}}
// CHECK:       `-CXXRecordDecl 0x{{.+}} prev 0x{{.+}} <col:24, col:57> col:31 struct TestTemplateDefaultType definition{{$}}
// CHECK:         |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init{{$}}
// CHECK:         | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr{{$}}
// CHECK:         | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
// CHECK:         | |-MoveConstructor exists simple trivial needs_implicit{{$}}
// CHECK:         | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
// CHECK:         | |-MoveAssignment exists simple trivial needs_implicit{{$}}
// CHECK:         | `-Destructor simple irrelevant trivial needs_implicit{{$}}
// CHECK:         `-CXXRecordDecl 0x{{.+}} <col:24, col:31> col:31 implicit struct TestTemplateDefaultType{{$}}

// CHECK:       ClassTemplateDecl 0x{{.+}} <{{.+}}:[[@LINE-148]]:3, col:31> col:31 TestTemplateDefaultNonType{{$}}
// CHECK:       |-NonTypeTemplateParmDecl 0x{{.+}} <col:12, col:20> col:16 'int' depth 0 index 0 I{{$}}
// CHECK:       | `-TemplateArgument <col:20> expr '42'{{$}}
// CHECK:       |   `-IntegerLiteral 0x{{.+}} <col:20> 'int' 42{{$}}
// CHECK:       `-CXXRecordDecl 0x{{.+}} <col:24, col:31> col:31 struct TestTemplateDefaultNonType{{$}}

// CHECK:       ClassTemplateDecl 0x{{.+}} <{{.+}}:{{.*}}:3, col:68> col:68 TestTemplateTemplateDefaultType{{$}}
// CHECK:       |-TemplateTemplateParmDecl 0x{{.+}} <col:12, col:42> col:37 depth 0 index 0 TT{{$}}
// CHECK:       | |-TemplateTypeParmDecl 0x{{.+}} <col:21> col:29 typename depth 1 index 0{{$}}
// CHECK:       | `-TemplateArgument <col:42> template 'TestClassTemplate':'testClassTemplateDecl::TestClassTemplate' qualified{{$}}
// CHECK:       |   `-ClassTemplateDecl 0x{{.+}} <line:{{.+}}:3, line:{{.+}}:3> line:{{.+}}:30 TestClassTemplate{{$}}
// CHECK:       `-CXXRecordDecl 0x{{.+}} <line:{{.*}}:61, col:68> col:68 struct TestTemplateTemplateDefaultType{{$}}

// CHECK:       ClassTemplateDecl 0x{{.+}} prev 0x{{.+}} <{{.+}}:{{.*}}:3, col:82> col:48 TestTemplateTemplateDefaultType{{$}}
// CHECK:       |-TemplateTemplateParmDecl 0x{{.+}} <col:12, col:37> col:37 depth 0 index 0 TT{{$}}
// CHECK:       | |-TemplateTypeParmDecl 0x{{.+}} <col:21> col:29 typename depth 1 index 0{{$}}
// CHECK:       | `-TemplateArgument <line:{{.*}}:42> template 'TestClassTemplate':'testClassTemplateDecl::TestClassTemplate' qualified{{$}}
// CHECK:       |   |-inherited from TemplateTemplateParm 0x{{.+}} 'TT'{{$}}
// CHECK:       |   `-ClassTemplateDecl 0x{{.+}} <line:{{.+}}:3, line:{{.+}}:3> line:{{.+}}:30 TestClassTemplate
// CHECK:       `-CXXRecordDecl 0x{{.+}} prev 0x{{.+}} <line:{{.*}}:41, col:82> col:48 struct TestTemplateTemplateDefaultType definition{{$}}
// CHECK:         |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init{{$}}
// CHECK:         | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr{{$}}
// CHECK:         | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
// CHECK:         | |-MoveConstructor exists simple trivial needs_implicit{{$}}
// CHECK:         | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
// CHECK:         | |-MoveAssignment exists simple trivial needs_implicit{{$}}
// CHECK:         | `-Destructor simple irrelevant trivial needs_implicit{{$}}
// CHECK:         `-CXXRecordDecl 0x{{.+}} <col:41, col:48> col:48 implicit struct TestTemplateTemplateDefaultType{{$}}


// PR15220 dump instantiation only once
namespace testCanonicalTemplate {
  class A {};

  template<typename T> void TestFunctionTemplate(T);
  template<typename T> void TestFunctionTemplate(T);
  void bar(A a) { TestFunctionTemplate(a); }
  // CHECK:      FunctionTemplateDecl 0x{{.+}} <{{.+}}:[[@LINE-3]]:3, col:51> col:29 TestFunctionTemplate{{$}}
  // CHECK:        |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 referenced typename depth 0 index 0 T{{$}}
  // CHECK:        |-FunctionDecl 0x{{.*}} <col:24, col:51> col:29 TestFunctionTemplate 'void (T)'{{$}}
  // CHECK:        | `-ParmVarDecl 0x{{.*}} <col:50> col:51 'T'{{$}}
  // CHECK:        `-FunctionDecl 0x{{.*}} <line:[[@LINE-6]]:24, col:51> col:29 used TestFunctionTemplate 'void (testCanonicalTemplate::A)' implicit_instantiation{{$}}
  // CHECK:          |-TemplateArgument type 'testCanonicalTemplate::A'{{$}}
  // CHECK:          | `-typeDetails: RecordType 0x{{.+}} 'testCanonicalTemplate::A'{{$}}
  // CHECK:          |   `-CXXRecord 0x{{.+}} 'A'{{$}}
  // CHECK:          `-ParmVarDecl 0x{{.*}} <col:50> col:51 'testCanonicalTemplate::A'{{$}}

  // CHECK:      FunctionTemplateDecl 0x{{.+}} prev 0x{{.+}} <{{.+}}:[[@LINE-12]]:3, col:51> col:29 TestFunctionTemplate{{$}}
  // CHECK:        |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 referenced typename depth 0 index 0 T{{$}}
  // CHECK:        |-FunctionDecl{{.*}} 0x{{.+}} prev 0x{{.+}} <col:24, col:51> col:29 TestFunctionTemplate 'void (T)'{{$}}
  // CHECK:        | `-ParmVarDecl 0x{{.+}} <col:50> col:51 'T'{{$}}
  // CHECK:        `-Function 0x{{.+}} 'TestFunctionTemplate' 'void (testCanonicalTemplate::A)'{{$}}
  // CHECK-NOT:      TemplateArgument{{$}}

  template<typename T1> class TestClassTemplate {
    template<typename T2> friend class TestClassTemplate;
  };
  TestClassTemplate<A> a;
  // CHECK:      ClassTemplateDecl 0x{{.+}} <{{.+}}:[[@LINE-4]]:3, line:[[@LINE-2]]:3> line:[[@LINE-4]]:31 TestClassTemplate{{$}}
  // CHECK:      |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 typename depth 0 index 0 T1{{$}}
  // CHECK:      |-CXXRecordDecl 0x{{.+}} <col:25, line:[[@LINE-4]]:3> line:[[@LINE-6]]:31 class TestClassTemplate definition{{$}}
  // CHECK:      | |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init{{$}}
  // CHECK:      | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr{{$}}
  // CHECK:      | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
  // CHECK:      | | |-MoveConstructor exists simple trivial needs_implicit{{$}}
  // CHECK:      | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
  // CHECK:      | | |-MoveAssignment exists simple trivial needs_implicit{{$}}
  // CHECK:      | | `-Destructor simple irrelevant trivial needs_implicit{{$}}
  // CHECK:      | |-CXXRecordDecl 0x{{.+}} <col:25, col:31> col:31 implicit class TestClassTemplate{{$}}
  // CHECK:      | `-FriendDecl 0x{{.+}} <line:[[@LINE-14]]:5, col:40> col:40{{$}}
  // CHECK:      |   `-ClassTemplateDecl 0x{{.+}} parent 0x{{.+}} <col:5, col:40> col:40 friend_undeclared TestClassTemplate{{$}}
  // CHECK:      |     |-TemplateTypeParmDecl 0x{{.+}} <col:14, col:23> col:23 typename depth 1 index 0 T2{{$}}
  // CHECK:      |     `-CXXRecordDecl 0x{{.+}} parent 0x{{.+}} <col:34, col:40> col:40 class TestClassTemplate{{$}}
  // CHECK:      `-ClassTemplateSpecializationDecl 0x{{.+}} <line:[[@LINE-19]]:3, line:[[@LINE-17]]:3> line:[[@LINE-19]]:31 class TestClassTemplate definition implicit_instantiation{{$}}
  // CHECK:        |-DefinitionData pass_in_registers empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init{{$}}
  // CHECK:        | |-DefaultConstructor exists trivial constexpr defaulted_is_constexpr{{$}}
  // CHECK:        | |-CopyConstructor simple trivial has_const_param implicit_has_const_param{{$}}
  // CHECK:        | |-MoveConstructor exists simple trivial{{$}}
  // CHECK:        | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param{{$}}
  // CHECK:        | |-MoveAssignment exists simple trivial needs_implicit{{$}}
  // CHECK:        | `-Destructor simple irrelevant trivial needs_implicit{{$}}
  // CHECK:        |-TemplateArgument type 'testCanonicalTemplate::A'{{$}}
  // CHECK:        | `-typeDetails: RecordType 0x{{.+}} 'testCanonicalTemplate::A'{{$}}
  // CHECK:        |   `-CXXRecord 0x{{.+}} 'A'{{$}}
  // CHECK:        |-CXXRecordDecl 0x{{.+}} <col:25, col:31> col:31 implicit class TestClassTemplate{{$}}
  // CHECK:        |-FriendDecl 0x{{.+}} <line:[[@LINE-30]]:5, col:40> col:40{{$}}
  // CHECK:        | `-ClassTemplateDecl 0x{{.+}} parent 0x{{.+}} prev 0x{{.+}} <col:5, col:40> col:40 friend TestClassTemplate{{$}}
  // CHECK:        |   |-TemplateTypeParmDecl 0x{{.+}} <col:14, col:23> col:23 typename depth 0 index 0 T2{{$}}
  // CHECK:        |   |-CXXRecordDecl 0x{{.+}} parent 0x{{.+}} prev 0x{{.+}} <col:34, col:40> col:40 class TestClassTemplate{{$}}
  // CHECK:        |   `-ClassTemplateSpecialization 0x{{.+}} 'TestClassTemplate'{{$}}
  // CHECK:        |-CXXConstructorDecl 0x{{.+}} <line:[[@LINE-36]]:31> col:31 implicit used constexpr TestClassTemplate 'void () noexcept' inline default trivial{{$}}
  // CHECK:        | `-CompoundStmt 0x{{.+}} <col:31>{{$}}
  // CHECK:        |-CXXConstructorDecl 0x{{.+}} implicit constexpr TestClassTemplate 'void (const TestClassTemplate<testCanonicalTemplate::A> &)' inline default trivial noexcept-unevaluated 0x{{.+}}{{$}}
  // CHECK:        | `-ParmVarDecl 0x{{.+}} 'const TestClassTemplate<testCanonicalTemplate::A> &'
  // CHECK:        CXXConstructorDecl {{.*}}
  // CHECK:        ParmVarDecl 0x{{.+}}


  template<typename T1> class TestClassTemplate2;
  template<typename T1> class TestClassTemplate2;
  template<typename T1> class TestClassTemplate2 {
  };
  TestClassTemplate2<A> a2;


//CHECK: ClassTemplateDecl {{.*}} TestClassTemplate2
//CHECK: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 0 T1
//CHECK: |-CXXRecordDecl {{.*}} class TestClassTemplate2
//CHECK: `-ClassTemplateSpecializationDecl {{.*}} class TestClassTemplate2 definition implicit_instantiation
//CHECK:   |-DefinitionData pass_in_registers empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
//CHECK:   | |-DefaultConstructor exists trivial constexpr defaulted_is_constexpr
//CHECK:   | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
//CHECK:   | |-MoveConstructor exists simple trivial
//CHECK:   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
//CHECK:   | |-MoveAssignment exists simple trivial needs_implicit
//CHECK:   | `-Destructor simple irrelevant trivial needs_implicit
//CHECK:   |-TemplateArgument type 'testCanonicalTemplate::A'
//CHECK:   | `-typeDetails: RecordType {{.*}} 'testCanonicalTemplate::A'
//CHECK:   |   `-CXXRecord {{.*}} 'A'
//CHECK:   |-CXXRecordDecl {{.*}} implicit class TestClassTemplate2
//CHECK:   |-CXXConstructorDecl {{.*}} implicit used constexpr TestClassTemplate2 'void () noexcept' inline default trivial
//CHECK:   | `-CompoundStmt {{.*}}
//CHECK:   |-CXXConstructorDecl {{.*}} implicit constexpr TestClassTemplate2 'void (const TestClassTemplate2<testCanonicalTemplate::A> &)' inline default trivial noexcept-unevaluated {{.*}}
//CHECK:   | `-ParmVarDecl {{.*}} 'const TestClassTemplate2<testCanonicalTemplate::A> &'
//CHECK:   |   `-typeDetails: LValueReferenceType {{.*}} 'const TestClassTemplate2<testCanonicalTemplate::A> &'
//CHECK:   |     `-qualTypeDetail: QualType {{.*}} 'const TestClassTemplate2<testCanonicalTemplate::A>' const
//CHECK:   |       `-typeDetails: ElaboratedType {{.*}} 'TestClassTemplate2<testCanonicalTemplate::A>' sugar
//CHECK:   |         `-typeDetails: RecordType {{.*}} 'testCanonicalTemplate::TestClassTemplate2<testCanonicalTemplate::A>'
//CHECK:   |           `-ClassTemplateSpecialization {{.*}} 'TestClassTemplate2'
//CHECK:   `-CXXConstructorDecl {{.*}} implicit constexpr TestClassTemplate2 'void (TestClassTemplate2<testCanonicalTemplate::A> &&)' inline default trivial noexcept-unevaluated {{.*}}
//CHECK:     `-ParmVarDecl {{.*}} 'TestClassTemplate2<testCanonicalTemplate::A> &&'
//CHECK:       `-typeDetails: RValueReferenceType {{.*}} 'TestClassTemplate2<testCanonicalTemplate::A> &&'
//CHECK:         `-typeDetails: ElaboratedType {{.*}} 'TestClassTemplate2<testCanonicalTemplate::A>' sugar
//CHECK:           `-typeDetails: RecordType {{.*}} 'testCanonicalTemplate::TestClassTemplate2<testCanonicalTemplate::A>'
//CHECK:             `-ClassTemplateSpecialization {{.*}} 'TestClassTemplate2'
//CHECK: ClassTemplateDecl {{.*}} prev {{.*}} TestClassTemplate2
//CHECK: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 0 T1
//CHECK: |-CXXRecordDecl {{.*}} prev {{.*}} class TestClassTemplate2
//CHECK: `-ClassTemplateSpecialization {{.*}} 'TestClassTemplate2'
//CHECK: ClassTemplateDecl {{.*}} prev {{.*}} TestClassTemplate2
//CHECK: |-TemplateTypeParmDecl {{.*}} typename depth 0 index 0 T1
//CHECK: |-CXXRecordDecl {{.*}} prev {{.*}} class TestClassTemplate2 definition
//CHECK: | |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
//CHECK: | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
//CHECK: | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
//CHECK: | | |-MoveConstructor exists simple trivial needs_implicit
//CHECK: | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
//CHECK: | | |-MoveAssignment exists simple trivial needs_implicit
//CHECK: | | `-Destructor simple irrelevant trivial needs_implicit
//CHECK: | `-CXXRecordDecl {{.*}} implicit class TestClassTemplate2
//CHECK: `-ClassTemplateSpecialization {{.*}} 'TestClassTemplate2'

  struct S {
      template<typename T> static const T TestVarTemplate; // declaration of a static data member template
  };
  template<typename T>
  const T S::TestVarTemplate = { }; // definition of a static data member template

  void f()
  {
    int i = S::TestVarTemplate<int>;
    int j = S::TestVarTemplate<int>;
  }

  // CHECK: VarTemplateDecl 0x{{.+}} <{{.+}}> col:43 TestVarTemplate
  // CHECK: |-TemplateTypeParmDecl 0x{{.+}} <{{.+}}> col:25 referenced typename depth 0 index 0 T
  // CHECK: |-VarDecl 0x{{.+}} <{{.+}}> col:43 TestVarTemplate 'const T' static
  // CHECK: | `-qualTypeDetail: QualType 0x{{.+}} 'const T' const
  // CHECK: |   `-typeDetails: TemplateTypeParmType 0x{{.+}} 'T' dependent depth 0 index 0
  // CHECK: |     `-TemplateTypeParm 0x{{.+}} 'T'
  // CHECK: | |-NestedNameSpecifier TypeSpec 'testCanonicalTemplate::S'
  // CHECK: | |-TemplateArgument type 'int'
  // CHECK: | | `-typeDetails: BuiltinType 0x{{.+}} 'int'
  // CHECK: | |-InitListExpr 0x{{.+}} <{{.+}}> 'int'
  // CHECK: | `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
  // CHECK: |   `-typeDetails: SubstTemplateTypeParmType {{.*}} 'int' sugar typename depth 0 index 0 T
  // CHECK: |     `-typeDetails: BuiltinType 0x{{.+}} 'int'
  // CHECK: `-VarTemplateSpecializationDecl 0x{{.+}} <{{.+}}> col:43 referenced TestVarTemplate 'const int' implicit_instantiation static
  // CHECK:   |-TemplateArgument type 'int'
  // CHECK:   | `-typeDetails: BuiltinType 0x{{.+}} 'int'
  // CHECK:   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
  // CHECK:     `-typeDetails: SubstTemplateTypeParmType 0x{{.+}} 'int' sugar typename depth 0 index 0 T
  // CHECK:       |-VarTemplate 0x{{.+}} 'TestVarTemplate'
  // CHECK:       `-typeDetails: BuiltinType 0x{{.+}} 'int'
  // CHECK: VarTemplateSpecializationDecl 0x{{.+}} <{{.+}}> col:43 referenced TestVarTemplate 'const int' implicit_instantiation static
  // CHECK: |-TemplateArgument type 'int'
  // CHECK: | `-typeDetails: BuiltinType 0x{{.+}} 'int'
  // CHECK: `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
  // CHECK:   `-typeDetails: SubstTemplateTypeParmType 0x{{.+}} 'int' sugar typename depth 0 index 0 T
  // CHECK:     |-VarTemplate 0x{{.+}} 'TestVarTemplate'
  // CHECK:     `-typeDetails: BuiltinType 0x{{.+}} 'int'
  // CHECK: VarTemplateDecl 0x{{.+}} parent 0x{{.+}} prev 0x{{.+}} <{{.+}}> col:14 TestVarTemplate
  // CHECK: |-TemplateTypeParmDecl 0x{{.+}} <{{.+}}> col:21 referenced typename depth 0 index 0 T
  // CHECK: |-VarDecl 0x{{.+}} parent 0x{{.+}} prev 0x{{.+}} <{{.+}}> col:14 TestVarTemplate 'const T' cinit
  // CHECK: | |-NestedNameSpecifier TypeSpec 'testCanonicalTemplate::S'
  // CHECK: | |-InitListExpr 0x{{.+}} <{{.+}}> 'void'
  // CHECK: | `-qualTypeDetail: QualType 0x{{.+}} 'const T' const
  // CHECK: |   `-typeDetails: TemplateTypeParmType 0x{{.+}} 'T' dependent depth 0 index 0
  // CHECK: |     `-TemplateTypeParm 0x{{.+}} 'T'
  // CHECK: |-VarTemplateSpecialization 0x{{.+}} 'TestVarTemplate' 'const int'
  // CHECK: `-VarTemplateSpecialization 0x{{.+}} 'TestVarTemplate' 'const int'
  // CHECK: VarTemplateSpecializationDecl 0x{{.+}} parent 0x{{.+}} prev 0x{{.+}} <{{.+}}> col:14 referenced TestVarTemplate 'const int' implicit_instantiation cinit
  // CHECK: |-NestedNameSpecifier TypeSpec 'testCanonicalTemplate::S'
  // CHECK: |-TemplateArgument type 'int'
  // CHECK: | `-typeDetails: BuiltinType 0x{{.+}} 'int'
  // CHECK: |-InitListExpr 0x{{.+}} <{{.+}}> 'int'
  // CHECK: `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
  // CHECK:   `-typeDetails: SubstTemplateTypeParmType 0x{{.+}} 'int' sugar typename depth 0 index 0 T
  // CHECK:     `-typeDetails: BuiltinType 0x{{.+}} 'int'
}

template <class T>
class TestClassScopeFunctionSpecialization {
  template<class U> void foo(U a) { }
  template<> void foo<int>(int a) { }
};
// CHECK:      FunctionTemplateDecl{{.*}} foo
// CHECK:        TemplateTypeParmDecl{{.*}} referenced class depth 1 index 0 U
// CHECK:        CXXMethodDecl{{.*}} foo 'void (U)' implicit-inline
// CHECK:          ParmVarDecl
// CHECK:          CompoundStmt
// CHECK:      CXXMethodDecl{{.*}} foo 'void (int)' explicit_specialization implicit-inline
// CHECK:        TemplateArgument{{.*}} 'int'
// CHECK:          BuiltinType{{.*}} 'int'
// CHECK:        ParmVarDecl
// CHECK:        CompoundStmt

namespace TestTemplateTypeParmDecl {
  template<typename ... T, class U = int> void foo();
}
// CHECK:      NamespaceDecl{{.*}} TestTemplateTypeParmDecl
// CHECK:        FunctionTemplateDecl
// CHECK:          TemplateTypeParmDecl{{.*}} typename depth 0 index 0 ... T
// CHECK:          TemplateTypeParmDecl{{.*}} class depth 0 index 1 U
// CHECK:            TemplateArgument type 'int'

namespace TestNonTypeTemplateParmDecl {
  template<int I = 1, int ... J> void foo();
}
// CHECK:      NamespaceDecl{{.*}} TestNonTypeTemplateParmDecl
// CHECK:        FunctionTemplateDecl
// CHECK:          NonTypeTemplateParmDecl{{.*}} 'int' depth 0 index 0 I
// CHECK:            TemplateArgument {{.*}} expr
// CHECK:              IntegerLiteral{{.*}} 'int' 1
// CHECK:          NonTypeTemplateParmDecl{{.*}} 'int' depth 0 index 1 ... J

namespace TestTemplateTemplateParmDecl {
  template<typename T> class A;
  template <template <typename> class T = A, template <typename> class ... U> void foo();
}
// CHECK:      NamespaceDecl{{.*}} TestTemplateTemplateParmDecl
// CHECK:        FunctionTemplateDecl
// CHECK:          TemplateTemplateParmDecl{{.*}} T
// CHECK:            TemplateTypeParmDecl{{.*}} typename
// CHECK:            TemplateArgument{{.*}} template 'A':'TestTemplateTemplateParmDecl::A' qualified{{$}}
// CHECK:              ClassTemplateDecl {{.*}} A
// CHECK:          TemplateTemplateParmDecl{{.*}} ... U
// CHECK:            TemplateTypeParmDecl{{.*}} typename

namespace TestTemplateArgument {
  template<typename> class A { };
  template<template<typename> class ...> class B { };
  int foo();

  template<typename> class testType { };
  template class testType<int>;
  // CHECK:      ClassTemplateSpecializationDecl{{.*}} class testType
  // CHECK:        TemplateArgument{{.*}} type 'int'

  template<int fp(void)> class testDecl { };
  template class testDecl<foo>;
  // CHECK:      ClassTemplateSpecializationDecl{{.*}} class testDecl
  // CHECK:        TemplateArgument{{.*}} decl
  // CHECK:          Function{{.*}}foo

  template class testDecl<nullptr>;
  // CHECK:      ClassTemplateSpecializationDecl{{.*}} class testDecl
  // CHECK:        TemplateArgument{{.*}} nullptr

  template<int> class testIntegral { };
  template class testIntegral<1>;
  // CHECK:      ClassTemplateSpecializationDecl{{.*}} class testIntegral
  // CHECK:        TemplateArgument{{.*}} integral '1'

  template<template<typename> class> class testTemplate { };
  template class testTemplate<A>;
  // CHECK:      ClassTemplateSpecializationDecl{{.*}} class testTemplate
  // CHECK:        TemplateArgument{{.*}} 'TestTemplateArgument::A'{{$}}

  template<template<typename> class ...T> class C {
    B<T...> testTemplateExpansion;
  };
  // FIXME: Need TemplateSpecializationType dumping to test TemplateExpansion.

  template<int, int = 0> class testExpr;
  template<int I> class testExpr<I> { };
  // CHECK:      ClassTemplatePartialSpecializationDecl{{.*}} class testExpr
  // CHECK:        TemplateArgument{{.*}} expr
  // CHECK:          DeclRefExpr{{.*}}I

  template<int, int ...> class testPack { };
  template class testPack<0, 1, 2>;
  // CHECK:      ClassTemplateSpecializationDecl{{.*}} class testPack
  // CHECK:        TemplateArgument{{.*}} integral '0'
  // CHECK:        TemplateArgument{{.*}} pack
  // CHECK:          TemplateArgument{{.*}} integral '1'
  // CHECK:          TemplateArgument{{.*}} integral '2'
}

namespace testUsingDecl {
  int i;
}
namespace TestUsingDecl {
  using testUsingDecl::i;
}
// CHECK:      NamespaceDecl{{.*}} TestUsingDecl
// CHECK:        UsingDecl{{.*}} testUsingDecl::i
// CHECK:        | `-NestedNameSpecifier Namespace 0x{{.*}} 'testUsingDecl
// CHECK:        UsingShadowDecl{{.*}} Var{{.*}} 'i' 'int'

namespace testUnresolvedUsing {
  class A { };
  template<class T> class B {
  public:
    A a;
  };
  template<class T> class TestUnresolvedUsing : public B<T> {
    using typename B<T>::a;
    using B<T>::a;
  };
}
// CHECK: CXXRecordDecl{{.*}} TestUnresolvedUsing
// CHECK:   UnresolvedUsingTypenameDecl{{.*}} B<T>::a
// CHECK:   UnresolvedUsingValueDecl{{.*}} B<T>::a

namespace TestLinkageSpecDecl {
  extern "C" void test1();
  extern "C++" void test2();
}
// CHECK:      NamespaceDecl{{.*}} TestLinkageSpecDecl
// CHECK:        LinkageSpecDecl{{.*}} C
// CHECK:          FunctionDecl
// CHECK:        LinkageSpecDecl{{.*}} C++
// CHECK:          FunctionDecl

class TestAccessSpecDecl {
public:
private:
protected:
};
// CHECK:      CXXRecordDecl{{.*}} class TestAccessSpecDecl
// CHECK:         CXXRecordDecl{{.*}} class TestAccessSpecDecl
// CHECK:         AccessSpecDecl{{.*}} public
// CHECK:         AccessSpecDecl{{.*}} private
// CHECK:         AccessSpecDecl{{.*}} protected

template<typename T> class TestFriendDecl {
  friend int foo();
  friend class A;
  friend T;
};
// CHECK:      CXXRecord{{.*}} TestFriendDecl
// CHECK:        CXXRecord{{.*}} TestFriendDecl
// CHECK:        FriendDecl
// CHECK:          FunctionDecl{{.*}} foo
// CHECK:        FriendDecl{{.*}} 'class A':'A'
// CHECK:          CXXRecordDecl{{.*}} class A
// CHECK:        FriendDecl{{.*}} 'T'

namespace TestFileScopeAsmDecl {
  asm("ret");
}
// CHECK:      NamespaceDecl{{.*}} TestFileScopeAsmDecl{{$}}
// CHECK:        FileScopeAsmDecl{{.*> .*$}}
// CHECK:          StringLiteral

namespace TestFriendDecl2 {
  void f();
  struct S {
    friend void f();
  };
}
// CHECK: NamespaceDecl [[TestFriendDecl2:0x.*]] <{{.*}}> {{.*}} TestFriendDecl2
// CHECK: |-FunctionDecl [[TestFriendDecl2_f:0x.*]] <{{.*}}> {{.*}} f 'void ()'
// CHECK: `-CXXRecordDecl {{.*}} struct S
// CHECK:   |-CXXRecordDecl {{.*}} struct S
// CHECK:   `-FriendDecl
// CHECK:     `-FunctionDecl {{.*}} parent [[TestFriendDecl2]] prev [[TestFriendDecl2_f]] <{{.*}}> {{.*}} f 'void ()'

namespace Comment {
  extern int Test;
  /// Something here.
  extern int Test;
  extern int Test;
}

// CHECK: VarDecl {{.*}} Test 'int' extern
// CHECK-NOT: FullComment
// CHECK: VarDecl {{.*}} Test 'int' extern
// CHECK: `-FullComment
// CHECK:   `-ParagraphComment
// CHECK:       `-TextComment
// CHECK: VarDecl {{.*}} Test 'int' extern
// CHECK-NOT: FullComment

namespace TestConstexprVariableTemplateWithInitializer {
  template<typename T> constexpr T foo{};
  // CHECK:      |-VarTemplateDecl 0x{{.+}} <{{.+}}> col:36 foo
  // CHECK-NEXT: | |-TemplateTypeParmDecl 0x{{.+}} <{{.+}}> col:21 referenced typename depth 0 index 0 T
  // CHECK-NEXT: | `-VarDecl 0x{{.+}} <{{.+}}> col:36 foo 'const T' constexpr listinit
  // CHECK-NEXT: |   |-InitListExpr 0x{{.+}} <{{.+}}> 'void'
  // CHECK-NEXT: |   `-qualTypeDetail: QualType 0x{{.+}} 'const T' const
  // CHECK-NEXT: |     `-typeDetails: TemplateTypeParmType 0x{{.+}} 'T' dependent depth 0 index 0
  // CHECK-NEXT: |       `-TemplateTypeParm 0x{{.+}} 'T'

  template<typename T> constexpr int val{42};
  // CHECK:      |-VarTemplateDecl 0x{{.+}} <{{.+}}> col:38 val
  // CHECK-NEXT: | |-TemplateTypeParmDecl 0x{{.+}} <{{.+}}> col:21 typename depth 0 index 0 T
  // CHECK-NEXT: | `-VarDecl 0x{{.+}} <{{.+}}> col:38 val 'const int' constexpr listinit
  // CHECK-NEXT: |   |-value: Int 42
  // CHECK-NEXT: |   |-InitListExpr 0x{{.+}} <{{.+}}> 'int'
  // CHECK-NEXT: |   | `-IntegerLiteral 0x{{.+}} <{{.+}}> 'int' 42
  // CHECK-NEXT: |   `-qualTypeDetail: QualType 0x{{.+}} 'const int' const
  // CHECK-NEXT: |     `-typeDetails: BuiltinType 0x{{.+}} 'int'

  template <typename _Tp>
  struct in_place_type_t {
    explicit in_place_type_t() = default;
  };

  template <typename _Tp>
  inline constexpr in_place_type_t<_Tp> in_place_type{};
  // CHECK:      |-VarTemplateDecl 0x{{.+}} <{{.+}}> col:41 in_place_type
  // CHECK-NEXT: | |-TemplateTypeParmDecl 0x{{.+}} <{{.+}}> col:22 referenced typename depth 0 index 0 _Tp
  // CHECK-NEXT: | `-VarDecl 0x{{.+}} <{{.+}}> col:41 in_place_type 'const in_place_type_t<{{.+}}>' inline constexpr listinit
  // CHECK-NEXT: |   |-InitListExpr 0x{{.+}} <{{.+}}> 'void'
  // CHECK-NEXT: |   `-qualTypeDetail: QualType 0x{{.+}} 'const in_place_type_t<{{.+}}>' const
  // CHECK-NEXT: |     `-typeDetails: ElaboratedType 0x{{.+}} 'in_place_type_t<{{.+}}>' sugar dependent
  // CHECK-NEXT: |       `-typeDetails: TemplateSpecializationType 0x{{.+}} 'in_place_type_t<{{.+}}>' dependent

  template <typename T> constexpr T call_init(0);
  // CHECK:      `-VarTemplateDecl 0x{{.+}} <{{.+}}> col:37 call_init
  // CHECK-NEXT:   |-TemplateTypeParmDecl 0x{{.+}} <{{.+}}> col:22 referenced typename depth 0 index 0 T
  // CHECK-NEXT:   `-VarDecl 0x{{.+}} <{{.+}}> col:37 call_init 'const T' constexpr callinit
  // CHECK-NEXT:     |-ParenListExpr 0x{{.+}} <{{.+}}> 'NULL TYPE'
  // CHECK-NEXT:     | `-IntegerLiteral 0x{{.+}} <{{.+}}> 'int' 0
  // CHECK-NEXT:     `-qualTypeDetail: QualType 0x{{.+}} 'const T' const
  // CHECK-NEXT:       `-typeDetails: TemplateTypeParmType 0x{{.+}} 'T' dependent depth 0 index 0
  // CHECK-NEXT:         `-TemplateTypeParm 0x{{.+}} 'T'

}
