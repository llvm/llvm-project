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
// CHECK: EnumDecl{{.*}} TestEnumDeclFixed 'int'

class testFieldDecl {
  int TestFieldDeclInit = 0;
};
// CHECK:      FieldDecl{{.*}} TestFieldDeclInit 'int'
// CHECK-NEXT:   IntegerLiteral

namespace testVarDeclNRVO {
  class A { };
  A foo() {
    A TestVarDeclNRVO;
    return TestVarDeclNRVO;
  }
}
// CHECK: VarDecl{{.*}} TestVarDeclNRVO 'A':'testVarDeclNRVO::A' nrvo

void testParmVarDeclInit(int TestParmVarDeclInit = 0);
// CHECK:      ParmVarDecl{{.*}} TestParmVarDeclInit 'int'
// CHECK-NEXT:   IntegerLiteral{{.*}}

namespace TestNamespaceDecl {
  int i;
}
// CHECK:      NamespaceDecl{{.*}} TestNamespaceDecl
// CHECK-NEXT:   VarDecl

namespace TestNamespaceDecl {
  int j;
}
// CHECK:      NamespaceDecl{{.*}} TestNamespaceDecl
// CHECK-NEXT:   original Namespace
// CHECK-NEXT:   VarDecl

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
// CHECK-NEXT:   UsingDirectiveDecl{{.*}} Namespace{{.*}} 'A'

namespace testNamespaceAlias {
  namespace A {
  }
}
namespace TestNamespaceAlias = testNamespaceAlias::A;
// CHECK:      NamespaceAliasDecl{{.*}} TestNamespaceAlias
// CHECK-NEXT:   Namespace{{.*}} 'A'

using TestTypeAliasDecl = int;
// CHECK: TypeAliasDecl{{.*}} TestTypeAliasDecl 'int'

namespace testTypeAliasTemplateDecl {
  template<typename T> class A;
  template<typename T> using TestTypeAliasTemplateDecl = A<T>;
}
// CHECK:      TypeAliasTemplateDecl{{.*}} TestTypeAliasTemplateDecl
// CHECK-NEXT:   TemplateTypeParmDecl
// CHECK-NEXT:   TypeAliasDecl{{.*}} TestTypeAliasTemplateDecl 'A<T>'

namespace testCXXRecordDecl {
  class TestEmpty {};
// CHECK:      CXXRecordDecl{{.*}} class TestEmpty
// CHECK-NEXT:   DefinitionData pass_in_registers empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK-NEXT:     DefaultConstructor exists trivial constexpr
// CHECK-NEXT:     CopyConstructor simple trivial has_const_param
// CHECK-NEXT:     MoveConstructor exists simple trivial
// CHECK-NEXT:     CopyAssignment simple trivial has_const_param
// CHECK-NEXT:     MoveAssignment exists simple trivial
// CHECK-NEXT:     Destructor simple irrelevant trivial

  class A { };
  class B { };
  class TestCXXRecordDecl : virtual A, public B {
    int i;
  };
}
// CHECK:      CXXRecordDecl{{.*}} class TestCXXRecordDecl
// CHECK-NEXT:   DefinitionData{{$}}
// CHECK-NEXT:     DefaultConstructor exists non_trivial
// CHECK-NEXT:     CopyConstructor simple non_trivial has_const_param
// CHECK-NEXT:     MoveConstructor exists simple non_trivial
// CHECK-NEXT:     CopyAssignment simple non_trivial has_const_param
// CHECK-NEXT:     MoveAssignment exists simple non_trivial
// CHECK-NEXT:     Destructor simple irrelevant trivial
// CHECK-NEXT:   virtual private 'A':'testCXXRecordDecl::A'
// CHECK-NEXT:   public 'B':'testCXXRecordDecl::B'
// CHECK-NEXT:   CXXRecordDecl{{.*}} class TestCXXRecordDecl
// CHECK-NEXT:   FieldDecl

template<class...T>
class TestCXXRecordDeclPack : public T... {
};
// CHECK:      CXXRecordDecl{{.*}} class TestCXXRecordDeclPack
// CHECK:        public 'T'...
// CHECK-NEXT:   CXXRecordDecl{{.*}} class TestCXXRecordDeclPack

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
// CHECK-NEXT:   ParmVarDecl{{.*}} a
// CHECK-NEXT:   ParmVarDecl{{.*}} i
// CHECK-NEXT:   CXXCtorInitializer{{.*}}A
// CHECK-NEXT:     Expr
// CHECK:        CXXCtorInitializer{{.*}}I
// CHECK-NEXT:     Expr
// CHECK:        CompoundStmt
// CHECK:      CXXConstructorDecl{{.*}} TestCXXConstructorDecl 'void {{.*}}'
// CHECK-NEXT:   ParmVarDecl{{.*}} a
// CHECK-NEXT:   CXXCtorInitializer{{.*}}TestCXXConstructorDecl
// CHECK-NEXT:     CXXConstructExpr{{.*}}TestCXXConstructorDecl

class TestCXXDestructorDecl {
  ~TestCXXDestructorDecl() { }
};
// CHECK:      CXXDestructorDecl{{.*}} ~TestCXXDestructorDecl 'void () noexcept'
// CHECK-NEXT:   CompoundStmt

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
// CHECK-NEXT:   CompoundStmt

namespace TestStaticAssertDecl {
  static_assert(true, "msg");
}
// CHECK:      NamespaceDecl{{.*}} TestStaticAssertDecl
// CHECK-NEXT:   StaticAssertDecl{{.*> .*$}}
// CHECK-NEXT:     CXXBoolLiteralExpr
// CHECK-NEXT:     StringLiteral

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
  // CHECK:       FunctionTemplateDecl 0x{{.+}} <{{.+}}:[[@LINE-14]]:3, col:55> col:29 TestFunctionTemplate
  // CHECK-NEXT:  |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 referenced typename depth 0 index 0 T
  // CHECK-NEXT:  |-FunctionDecl 0x{{.+}} <col:24, col:55> col:29 TestFunctionTemplate 'void (T)'
  // CHECK-NEXT:  | |-ParmVarDecl 0x{{.+}} <col:50> col:51 'T'
  // CHECK-NEXT:  | `-CompoundStmt 0x{{.+}} <col:53, col:55>
  // CHECK-NEXT:  |-FunctionDecl 0x{{.+}} <col:24, col:55> col:29 used TestFunctionTemplate 'void (testFunctionTemplateDecl::A)'
  // CHECK-NEXT:  | |-TemplateArgument type 'testFunctionTemplateDecl::A'
  // CHECK-NEXT:  | | `-RecordType 0{{.+}} 'testFunctionTemplateDecl::A'
  // CHECK-NEXT:  | |   `-CXXRecord 0x{{.+}} 'A'
  // CHECK-NEXT:  | |-ParmVarDecl 0x{{.+}} <col:50> col:51 'testFunctionTemplateDecl::A':'testFunctionTemplateDecl::A'
  // CHECK-NEXT:  | `-CompoundStmt 0x{{.+}} <col:53, col:55>
  // CHECK-NEXT:  |-Function 0x{{.+}} 'TestFunctionTemplate' 'void (B)'
  // CHECK-NEXT:  |-FunctionDecl 0x{{.+}} <col:24, col:55> col:29 TestFunctionTemplate 'void (testFunctionTemplateDecl::C)'
  // CHECK-NEXT:  | |-TemplateArgument type 'testFunctionTemplateDecl::C'
  // CHECK-NEXT:  | | `-RecordType 0{{.+}} 'testFunctionTemplateDecl::C'
  // CHECK-NEXT:  | |   `-CXXRecord 0x{{.+}} 'C'
  // CHECK-NEXT:  | `-ParmVarDecl 0x{{.+}} <col:50> col:51 'testFunctionTemplateDecl::C':'testFunctionTemplateDecl::C'
  // CHECK-NEXT:  `-FunctionDecl 0x{{.+}} <col:24, col:55> col:29 TestFunctionTemplate 'void (testFunctionTemplateDecl::D)'
  // CHECK-NEXT:    |-TemplateArgument type 'testFunctionTemplateDecl::D'
  // CHECK-NEXT:    | `-RecordType 0{{.+}} 'testFunctionTemplateDecl::D'
  // CHECK-NEXT:    |   `-CXXRecord 0x{{.+}} 'D'
  // CHECK-NEXT:    |-ParmVarDecl 0x{{.+}} <col:50> col:51 'testFunctionTemplateDecl::D':'testFunctionTemplateDecl::D'
  // CHECK-NEXT:    `-CompoundStmt 0x{{.+}} <col:53, col:55>

  // CHECK:       FunctionDecl 0x{{.+}} prev 0x{{.+}} <{{.+}}:[[@LINE-32]]:3, col:41> col:19 TestFunctionTemplate 'void (B)'
  // CHECK-NEXT:  |-TemplateArgument type 'testFunctionTemplateDecl::B'
  // CHECK-NEXT:  | `-RecordType 0{{.+}} 'testFunctionTemplateDecl::B'
  // CHECK-NEXT:  |   `-CXXRecord 0x{{.+}} 'B'
  // CHECK-NEXT:  `-ParmVarDecl 0x{{.+}} <col:40> col:41 'B':'testFunctionTemplateDecl::B'


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

// CHECK:       ClassTemplateDecl 0x{{.+}} <{{.+}}:[[@LINE-40]]:3, line:[[@LINE-34]]:3> line:[[@LINE-40]]:30 TestClassTemplate
// CHECK-NEXT:  |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 typename depth 0 index 0 T
// CHECK-NEXT:  |-CXXRecordDecl 0x{{.+}} <col:24, line:[[@LINE-36]]:3> line:[[@LINE-42]]:30 class TestClassTemplate definition
// CHECK-NEXT:  | |-DefinitionData standard_layout has_user_declared_ctor can_const_default_init
// CHECK-NEXT:  | | |-DefaultConstructor exists non_trivial user_provided
// CHECK-NEXT:  | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:  | | |-MoveConstructor
// CHECK-NEXT:  | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:  | | |-MoveAssignment
// CHECK-NEXT:  | | `-Destructor
// CHECK-NEXT:  | |-CXXRecordDecl 0x{{.+}} <col:24, col:30> col:30 implicit referenced class TestClassTemplate
// CHECK-NEXT:  | |-AccessSpecDecl 0x{{.+}} <line:[[@LINE-50]]:3, col:9> col:3 public
// CHECK-NEXT:  | |-CXXConstructorDecl 0x{{.+}} <line:[[@LINE-50]]:5, col:23> col:5 TestClassTemplate<T> 'void ()'
// CHECK-NEXT:  | |-CXXDestructorDecl 0x{{.+}} <line:[[@LINE-50]]:5, col:24> col:5 ~TestClassTemplate<T> 'void ()'
// CHECK-NEXT:  | |-CXXMethodDecl 0x{{.+}} <line:[[@LINE-50]]:5, col:11> col:9 j 'int ()'
// CHECK-NEXT:  | `-FieldDecl 0x{{.+}} <line:[[@LINE-50]]:5, col:9> col:9 i 'int'
// CHECK-NEXT:  |-ClassTemplateSpecializationDecl 0x{{.+}} <line:[[@LINE-56]]:3, line:[[@LINE-50]]:3> line:[[@LINE-56]]:30 class TestClassTemplate definition
// CHECK-NEXT:  | |-DefinitionData standard_layout has_user_declared_ctor can_const_default_init
// CHECK-NEXT:  | | |-DefaultConstructor exists non_trivial user_provided
// CHECK-NEXT:  | | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
// CHECK-NEXT:  | | |-MoveConstructor
// CHECK-NEXT:  | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:  | | |-MoveAssignment
// CHECK-NEXT:  | | `-Destructor non_trivial user_declared
// CHECK-NEXT:  | |-TemplateArgument type 'testClassTemplateDecl::A'
// CHECK-NEXT:  | | `-RecordType 0{{.+}} 'testClassTemplateDecl::A'
// CHECK-NEXT:  | |   `-CXXRecord 0x{{.+}} 'A'
// CHECK-NEXT:  | |-CXXRecordDecl 0x{{.+}} <col:24, col:30> col:30 implicit class TestClassTemplate
// CHECK-NEXT:  | |-AccessSpecDecl 0x{{.+}} <line:[[@LINE-67]]:3, col:9> col:3 public
// CHECK-NEXT:  | |-CXXConstructorDecl 0x{{.+}} <line:[[@LINE-67]]:5, col:23> col:5 used TestClassTemplate 'void ()'
// CHECK-NEXT:  | |-CXXDestructorDecl 0x{{.+}} <line:[[@LINE-67]]:5, col:24> col:5 used ~TestClassTemplate 'void () noexcept'
// CHECK-NEXT:  | |-CXXMethodDecl 0x{{.+}} <line:[[@LINE-67]]:5, col:11> col:9 j 'int ()'
// CHECK-NEXT:  | |-FieldDecl 0x{{.+}} <line:[[@LINE-67]]:5, col:9> col:9 i 'int'
// CHECK-NEXT:  | `-CXXConstructorDecl 0x{{.+}} <line:[[@LINE-73]]:30> col:30 implicit constexpr TestClassTemplate 'void (const TestClassTemplate<A> &)' inline default trivial noexcept-unevaluated 0x{{.+}}
// CHECK-NEXT:  |   `-ParmVarDecl 0x{{.+}} <col:30> col:30 'const TestClassTemplate<A> &'
// CHECK-NEXT:  |-ClassTemplateSpecialization 0x{{.+}} 'TestClassTemplate'
// CHECK-NEXT:  |-ClassTemplateSpecialization 0x{{.+}} 'TestClassTemplate'
// CHECK-NEXT:  `-ClassTemplateSpecialization 0x{{.+}} 'TestClassTemplate'

// CHECK:       ClassTemplateSpecializationDecl 0x{{.+}} <{{.+}}:[[@LINE-67]]:3, line:[[@LINE-65]]:3> line:[[@LINE-67]]:20 class TestClassTemplate definition
// CHECK-NEXT:  |-DefinitionData pass_in_registers standard_layout trivially_copyable trivial literal
// CHECK-NEXT:  | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:  | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:  | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:  | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:  | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:  | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:  |-TemplateArgument type 'testClassTemplateDecl::B'
// CHECK-NEXT:  | `-RecordType 0{{.+}} 'testClassTemplateDecl::B'
// CHECK-NEXT:  |   `-CXXRecord 0x{{.+}} 'B'
// CHECK-NEXT:  |-CXXRecordDecl 0x{{.+}} <col:14, col:20> col:20 implicit class TestClassTemplate
// CHECK-NEXT:  `-FieldDecl 0x{{.+}} <line:[[@LINE-78]]:5, col:9> col:9 j 'int'

// CHECK:       ClassTemplateSpecializationDecl 0x{{.+}} <{{.+}}:{{.*}}:3, col:44> col:25 class TestClassTemplate definition
// CHECK-NEXT:  |-DefinitionData standard_layout has_user_declared_ctor can_const_default_init
// CHECK-NEXT:  | |-DefaultConstructor exists non_trivial user_provided
// CHECK-NEXT:  | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:  | |-MoveConstructor
// CHECK-NEXT:  | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:  | |-MoveAssignment
// CHECK-NEXT:  | `-Destructor non_trivial user_declared
// CHECK-NEXT:  |-TemplateArgument type 'testClassTemplateDecl::C'
// CHECK-NEXT:  | `-RecordType 0{{.+}} 'testClassTemplateDecl::C'
// CHECK-NEXT:  |   `-CXXRecord 0x{{.+}} 'C'
// CHECK-NEXT:  |-CXXRecordDecl 0x{{.+}} <line:[[@LINE-104]]:24, col:30> col:30 implicit class TestClassTemplate
// CHECK-NEXT:  |-AccessSpecDecl 0x{{.+}} <line:[[@LINE-104]]:3, col:9> col:3 public
// CHECK-NEXT:  |-CXXConstructorDecl 0x{{.+}} <line:[[@LINE-104]]:5, col:23> col:5 TestClassTemplate 'void ()'
// CHECK-NEXT:  |-CXXDestructorDecl 0x{{.+}} <line:[[@LINE-104]]:5, col:24> col:5 ~TestClassTemplate 'void ()' noexcept-unevaluated 0x{{.+}}
// CHECK-NEXT:  |-CXXMethodDecl 0x{{.+}} <line:[[@LINE-104]]:5, col:11> col:9 j 'int ()'
// CHECK-NEXT:  `-FieldDecl 0x{{.+}} <line:[[@LINE-104]]:5, col:9> col:9 i 'int'

// CHECK:       ClassTemplateSpecializationDecl 0x{{.+}} <{{.+}}:[[@LINE-91]]:3, col:37> col:18 class TestClassTemplate definition
// CHECK-NEXT:  |-DefinitionData standard_layout has_user_declared_ctor can_const_default_init
// CHECK-NEXT:  | |-DefaultConstructor exists non_trivial user_provided
// CHECK-NEXT:  | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:  | |-MoveConstructor
// CHECK-NEXT:  | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:  | |-MoveAssignment
// CHECK-NEXT:  | `-Destructor non_trivial user_declared
// CHECK-NEXT:  |-TemplateArgument type 'testClassTemplateDecl::D'
// CHECK-NEXT:  | `-RecordType 0{{.+}} 'testClassTemplateDecl::D'
// CHECK-NEXT:  |   `-CXXRecord 0x{{.+}} 'D'
// CHECK-NEXT:  |-CXXRecordDecl 0x{{.+}} <line:[[@LINE-122]]:24, col:30> col:30 implicit class TestClassTemplate
// CHECK-NEXT:  |-AccessSpecDecl 0x{{.+}} <line:[[@LINE-122]]:3, col:9> col:3 public
// CHECK-NEXT:  |-CXXConstructorDecl 0x{{.+}} <line:[[@LINE-122]]:5, col:23> col:5 TestClassTemplate 'void ()'
// CHECK-NEXT:  |-CXXDestructorDecl 0x{{.+}} <line:[[@LINE-122]]:5, col:24> col:5 ~TestClassTemplate 'void ()' noexcept-unevaluated 0x{{.+}}
// CHECK-NEXT:  |-CXXMethodDecl 0x{{.+}} <line:[[@LINE-122]]:5, col:11> col:9 j 'int ()'
// CHECK-NEXT:  `-FieldDecl 0x{{.+}} <line:[[@LINE-122]]:5, col:9> col:9 i 'int'

// CHECK:      ClassTemplateDecl 0x{{.+}} <{{.+}}:[[@LINE-106]]:3, line:[[@LINE-104]]:3> line:[[@LINE-106]]:44 TestClassTemplatePartial
// CHECK-NEXT:  |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 typename depth 0 index 0 T1
// CHECK-NEXT:  |-TemplateTypeParmDecl 0x{{.+}} <col:25, col:34> col:34 typename depth 0 index 1 T2
// CHECK-NEXT:  `-CXXRecordDecl 0x{{.+}} <col:38, line:[[@LINE-107]]:3> line:[[@LINE-109]]:44 class TestClassTemplatePartial definition
// CHECK-NEXT:    |-DefinitionData standard_layout trivially_copyable trivial literal
// CHECK-NEXT:    | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:    | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:    | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:    | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:    | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:    | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:    |-CXXRecordDecl 0x{{.+}} <col:38, col:44> col:44 implicit class TestClassTemplatePartial
// CHECK-NEXT:    `-FieldDecl 0x{{.+}} <line:[[@LINE-117]]:5, col:9> col:9 i 'int'

// CHECK:       ClassTemplatePartialSpecializationDecl 0x{{.+}} <{{.+}}:[[@LINE-117]]:3, line:[[@LINE-115]]:3> line:[[@LINE-117]]:31 class TestClassTemplatePartial definition
// CHECK-NEXT:  |-DefinitionData standard_layout trivially_copyable trivial literal
// CHECK-NEXT:  | |-DefaultConstructor exists trivial needs_implicit
// CHECK-NEXT:  | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:  | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:  | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:  | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:  | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:  |-TemplateArgument type 'type-parameter-0-0'
// CHECK-NEXT:  | `-TemplateTypeParmType 0x{{.+}} 'type-parameter-0-0' dependent depth 0 index 0
// CHECK-NEXT:  |-TemplateArgument type 'testClassTemplateDecl::A'
// CHECK-NEXT:  | `-RecordType 0x{{.+}} 'testClassTemplateDecl::A'
// CHECK-NEXT:  |   `-CXXRecord 0x{{.+}} 'A'
// CHECK-NEXT:  |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 referenced typename depth 0 index 0 T1
// CHECK-NEXT:  |-CXXRecordDecl 0x{{.+}} <col:25, col:31> col:31 implicit class TestClassTemplatePartial
// CHECK-NEXT:  `-FieldDecl 0x{{.+}} <line:[[@LINE-131]]:5, col:9> col:9 j 'int'

// CHECK:       ClassTemplateDecl 0x{{.+}} <{{.+}}:[[@LINE-130]]:3, col:37> col:37 TestTemplateDefaultType
// CHECK-NEXT:  |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:25> col:21 typename depth 0 index 0 T
// CHECK-NEXT:  | `-TemplateArgument type 'int'
// CHECK-NEXT:  |   `-BuiltinType 0x{{.+}} 'int'
// CHECK-NEXT:  `-CXXRecordDecl 0x{{.+}} <col:30, col:37> col:37 struct TestTemplateDefaultType

// CHECK:       ClassTemplateDecl 0x{{.+}} prev 0x{{.+}} <{{.+}}:[[@LINE-135]]:3, col:57> col:31 TestTemplateDefaultType
// CHECK-NEXT:  |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 typename depth 0 index 0 T
// CHECK-NEXT:  | `-TemplateArgument type 'int'
// CHECK-NEXT:  |   |-inherited from TemplateTypeParm 0x{{.+}} 'T'
// CHECK-NEXT:  |   `-BuiltinType 0x{{.+}} 'int'
// CHECK-NEXT:  `-CXXRecordDecl 0x{{.+}} prev 0x{{.+}} <col:24, col:57> col:31 struct TestTemplateDefaultType definition
// CHECK-NEXT:    |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK-NEXT:    | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// CHECK-NEXT:    | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:    | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:    | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:    | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:    | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:    `-CXXRecordDecl 0x{{.+}} <col:24, col:31> col:31 implicit struct TestTemplateDefaultType

// CHECK:       ClassTemplateDecl 0x{{.+}} <{{.+}}:[[@LINE-148]]:3, col:31> col:31 TestTemplateDefaultNonType
// CHECK-NEXT:  |-NonTypeTemplateParmDecl 0x{{.+}} <col:12, col:20> col:16 'int' depth 0 index 0 I
// CHECK-NEXT:  | `-TemplateArgument expr
// CHECK-NEXT:  |   `-IntegerLiteral 0x{{.+}} <col:20> 'int' 42
// CHECK-NEXT:  `-CXXRecordDecl 0x{{.+}} <col:24, col:31> col:31 struct TestTemplateDefaultNonType

// CHECK:       ClassTemplateDecl 0x{{.+}} <{{.+}}:{{.*}}:3, col:68> col:68 TestTemplateTemplateDefaultType
// CHECK-NEXT:  |-TemplateTemplateParmDecl 0x{{.+}} <col:12, col:42> col:37 depth 0 index 0 TT
// CHECK-NEXT:  | |-TemplateTypeParmDecl 0x{{.+}} <col:21> col:29 typename depth 1 index 0
// CHECK-NEXT:  | `-TemplateArgument <col:42> template TestClassTemplate
// CHECK-NEXT:  `-CXXRecordDecl 0x{{.+}} <col:61, col:68> col:68 struct TestTemplateTemplateDefaultType

// CHECK:       ClassTemplateDecl 0x{{.+}} prev 0x{{.+}} <{{.+}}:{{.*}}:3, col:82> col:48 TestTemplateTemplateDefaultType
// CHECK-NEXT:  |-TemplateTemplateParmDecl 0x{{.+}} <col:12, col:37> col:37 depth 0 index 0 TT
// CHECK-NEXT:  | |-TemplateTypeParmDecl 0x{{.+}} <col:21> col:29 typename depth 1 index 0
// CHECK-NEXT:  | `-TemplateArgument <line:{{.*}}:42> template TestClassTemplate
// CHECK-NEXT:  |   `-inherited from TemplateTemplateParm 0x{{.+}} 'TT'
// CHECK-NEXT:  `-CXXRecordDecl 0x{{.+}} prev 0x{{.+}} <line:{{.*}}:41, col:82> col:48 struct TestTemplateTemplateDefaultType definition
// CHECK-NEXT:    |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
// CHECK-NEXT:    | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
// CHECK-NEXT:    | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:    | |-MoveConstructor exists simple trivial needs_implicit
// CHECK-NEXT:    | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
// CHECK-NEXT:    | |-MoveAssignment exists simple trivial needs_implicit
// CHECK-NEXT:    | `-Destructor simple irrelevant trivial needs_implicit
// CHECK-NEXT:    `-CXXRecordDecl 0x{{.+}} <col:41, col:48> col:48 implicit struct TestTemplateTemplateDefaultType


// PR15220 dump instantiation only once
namespace testCanonicalTemplate {
  class A {};

  template<typename T> void TestFunctionTemplate(T);
  template<typename T> void TestFunctionTemplate(T);
  void bar(A a) { TestFunctionTemplate(a); }
  // CHECK:      FunctionTemplateDecl 0x{{.+}} <{{.+}}:[[@LINE-3]]:3, col:51> col:29 TestFunctionTemplate
  // CHECK-NEXT:   |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 referenced typename depth 0 index 0 T
  // CHECK-NEXT:   |-FunctionDecl 0x{{.*}} <col:24, col:51> col:29 TestFunctionTemplate 'void (T)'
  // CHECK-NEXT:   | `-ParmVarDecl 0x{{.*}} <col:50> col:51 'T'
  // CHECK-NEXT:   `-FunctionDecl 0x{{.*}} <line:[[@LINE-6]]:24, col:51> col:29 used TestFunctionTemplate 'void (testCanonicalTemplate::A)'
  // CHECK-NEXT:     |-TemplateArgument type 'testCanonicalTemplate::A'
  // CHECK-NEXT:     | `-RecordType 0x{{.+}} 'testCanonicalTemplate::A'
  // CHECK-NEXT:     |   `-CXXRecord 0x{{.+}} 'A'
  // CHECK-NEXT:     `-ParmVarDecl 0x{{.*}} <col:50> col:51 'testCanonicalTemplate::A':'testCanonicalTemplate::A'

  // CHECK:      FunctionTemplateDecl 0x{{.+}} prev 0x{{.+}} <{{.+}}:[[@LINE-12]]:3, col:51> col:29 TestFunctionTemplate
  // CHECK-NEXT:   |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 referenced typename depth 0 index 0 T
  // CHECK-NEXT:   |-FunctionDecl{{.*}} 0x{{.+}} prev 0x{{.+}} <col:24, col:51> col:29 TestFunctionTemplate 'void (T)'
  // CHECK-NEXT:   | `-ParmVarDecl 0x{{.+}} <col:50> col:51 'T'
  // CHECK-NEXT:   `-Function 0x{{.+}} 'TestFunctionTemplate' 'void (testCanonicalTemplate::A)'
  // CHECK-NOT:      TemplateArgument

  template<typename T1> class TestClassTemplate {
    template<typename T2> friend class TestClassTemplate;
  };
  TestClassTemplate<A> a;
  // CHECK:      ClassTemplateDecl 0x{{.+}} <{{.+}}:[[@LINE-4]]:3, line:[[@LINE-2]]:3> line:[[@LINE-4]]:31 TestClassTemplate
  // CHECK-NEXT: |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 typename depth 0 index 0 T1
  // CHECK-NEXT: |-CXXRecordDecl 0x{{.+}} <col:25, line:[[@LINE-4]]:3> line:[[@LINE-6]]:31 class TestClassTemplate definition
  // CHECK-NEXT: | |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
  // CHECK-NEXT: | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
  // CHECK-NEXT: | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
  // CHECK-NEXT: | | |-MoveConstructor exists simple trivial needs_implicit
  // CHECK-NEXT: | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
  // CHECK-NEXT: | | |-MoveAssignment exists simple trivial needs_implicit
  // CHECK-NEXT: | | `-Destructor simple irrelevant trivial needs_implicit
  // CHECK-NEXT: | |-CXXRecordDecl 0x{{.+}} <col:25, col:31> col:31 implicit class TestClassTemplate
  // CHECK-NEXT: | `-FriendDecl 0x{{.+}} <line:[[@LINE-14]]:5, col:40> col:40
  // CHECK-NEXT: |   `-ClassTemplateDecl 0x{{.+}} parent 0x{{.+}} <col:5, col:40> col:40 TestClassTemplate
  // CHECK-NEXT: |     |-TemplateTypeParmDecl 0x{{.+}} <col:14, col:23> col:23 typename depth 1 index 0 T2
  // CHECK-NEXT: |     `-CXXRecordDecl 0x{{.+}} parent 0x{{.+}} <col:34, col:40> col:40 class TestClassTemplate
  // CHECK-NEXT: `-ClassTemplateSpecializationDecl 0x{{.+}} <line:[[@LINE-19]]:3, line:[[@LINE-17]]:3> line:[[@LINE-19]]:31 class TestClassTemplate definition
  // CHECK-NEXT:   |-DefinitionData pass_in_registers empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
  // CHECK-NEXT:   | |-DefaultConstructor exists trivial constexpr defaulted_is_constexpr
  // CHECK-NEXT:   | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
  // CHECK-NEXT:   | |-MoveConstructor exists simple trivial
  // CHECK-NEXT:   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
  // CHECK-NEXT:   | |-MoveAssignment exists simple trivial needs_implicit
  // CHECK-NEXT:   | `-Destructor simple irrelevant trivial needs_implicit
  // CHECK-NEXT:   |-TemplateArgument type 'testCanonicalTemplate::A'
  // CHECK-NEXT:   | `-RecordType 0x{{.+}} 'testCanonicalTemplate::A'
  // CHECK-NEXT:   |   `-CXXRecord 0x{{.+}} 'A'
  // CHECK-NEXT:   |-CXXRecordDecl 0x{{.+}} <col:25, col:31> col:31 implicit class TestClassTemplate
  // CHECK-NEXT:   |-FriendDecl 0x{{.+}} <line:[[@LINE-30]]:5, col:40> col:40
  // CHECK-NEXT:   | `-ClassTemplateDecl 0x{{.+}} parent 0x{{.+}} prev 0x{{.+}} <col:5, col:40> col:40 TestClassTemplate
  // CHECK-NEXT:   |   |-TemplateTypeParmDecl 0x{{.+}} <col:14, col:23> col:23 typename depth 0 index 0 T2
  // CHECK-NEXT:   |   |-CXXRecordDecl 0x{{.+}} parent 0x{{.+}} prev 0x{{.+}} <col:34, col:40> col:40 class TestClassTemplate
  // CHECK-NEXT:   |   `-ClassTemplateSpecialization 0x{{.+}} 'TestClassTemplate'
  // CHECK-NEXT:   |-CXXConstructorDecl 0x{{.+}} <line:[[@LINE-36]]:31> col:31 implicit used constexpr TestClassTemplate 'void () noexcept' inline default trivial
  // CHECK-NEXT:   | `-CompoundStmt 0x{{.+}} <col:31>
  // CHECK-NEXT:   |-CXXConstructorDecl 0x{{.+}} <col:31> col:31 implicit constexpr TestClassTemplate 'void (const TestClassTemplate<A> &)' inline default trivial noexcept-unevaluated 0x{{.+}}
  // CHECK-NEXT:   | `-ParmVarDecl 0x{{.+}} <col:31> col:31 'const TestClassTemplate<A> &'
  // CHECK-NEXT:   `-CXXConstructorDecl 0x{{.+}} <col:31> col:31 implicit constexpr TestClassTemplate 'void (TestClassTemplate<A> &&)' inline default trivial noexcept-unevaluated 0x{{.+}}
  // CHECK-NEXT:     `-ParmVarDecl 0x{{.+}} <col:31> col:31 'TestClassTemplate<A> &&'


  template<typename T1> class TestClassTemplate2;
  template<typename T1> class TestClassTemplate2;
  template<typename T1> class TestClassTemplate2 {
  };
  TestClassTemplate2<A> a2;
  // CHECK:      ClassTemplateDecl 0x{{.+}} <{{.+}}:[[@LINE-5]]:3, col:31> col:31 TestClassTemplate2
  // CHECK-NEXT: |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 typename depth 0 index 0 T1
  // CHECK-NEXT: |-CXXRecordDecl 0x{{.+}} <col:25, col:31> col:31 class TestClassTemplate2
  // CHECK-NEXT: `-ClassTemplateSpecializationDecl 0x{{.+}} <line:[[@LINE-6]]:3, line:[[@LINE-5]]:3> line:[[@LINE-6]]:31 class TestClassTemplate2 definition
  // CHECK-NEXT:   |-DefinitionData pass_in_registers empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
  // CHECK-NEXT:   | |-DefaultConstructor exists trivial constexpr defaulted_is_constexpr
  // CHECK-NEXT:   | |-CopyConstructor simple trivial has_const_param implicit_has_const_param
  // CHECK-NEXT:   | |-MoveConstructor exists simple trivial
  // CHECK-NEXT:   | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
  // CHECK-NEXT:   | |-MoveAssignment exists simple trivial needs_implicit
  // CHECK-NEXT:   | `-Destructor simple irrelevant trivial needs_implicit
  // CHECK-NEXT:   |-TemplateArgument type 'testCanonicalTemplate::A'
  // CHECK-NEXT:   | `-RecordType 0x{{.+}} 'testCanonicalTemplate::A'
  // CHECK-NEXT:   |   `-CXXRecord 0x{{.+}} 'A'
  // CHECK-NEXT:   |-CXXRecordDecl 0x{{.+}} <col:25, col:31> col:31 implicit class TestClassTemplate2
  // CHECK-NEXT:   |-CXXConstructorDecl 0x{{.+}} <col:31> col:31 implicit used constexpr TestClassTemplate2 'void () noexcept' inline default trivial
  // CHECK-NEXT:   | `-CompoundStmt 0x{{.+}} <col:31>
  // CHECK-NEXT:   |-CXXConstructorDecl 0x{{.+}} <col:31> col:31 implicit constexpr TestClassTemplate2 'void (const TestClassTemplate2<A> &)' inline default trivial noexcept-unevaluated 0x{{.+}}
  // CHECK-NEXT:   | `-ParmVarDecl 0x{{.+}} <col:31> col:31 'const TestClassTemplate2<A> &'
  // CHECK-NEXT:   `-CXXConstructorDecl 0x{{.+}} <col:31> col:31 implicit constexpr TestClassTemplate2 'void (TestClassTemplate2<A> &&)' inline default trivial noexcept-unevaluated 0x{{.+}}
  // CHECK-NEXT:     `-ParmVarDecl 0x{{.+}} <col:31> col:31 'TestClassTemplate2<A> &&'

  // CHECK:      ClassTemplateDecl 0x{{.+}} prev 0x{{.+}} <{{.+}}:[[@LINE-26]]:3, col:31> col:31 TestClassTemplate2
  // CHECK-NEXT: |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 typename depth 0 index 0 T1
  // CHECK-NEXT: |-CXXRecordDecl 0x{{.+}} prev 0x{{.+}} <col:25, col:31> col:31 class TestClassTemplate2
  // CHECK-NEXT: `-ClassTemplateSpecialization 0x{{.+}} 'TestClassTemplate2'

  // CHECK:      ClassTemplateDecl 0x{{.+}} prev 0x{{.+}} <{{.+}}:[[@LINE-30]]:3, line:[[@LINE-29]]:3> line:[[@LINE-30]]:31 TestClassTemplate2
  // CHECK-NEXT: |-TemplateTypeParmDecl 0x{{.+}} <col:12, col:21> col:21 typename depth 0 index 0 T1
  // CHECK-NEXT: |-CXXRecordDecl 0x{{.+}} prev 0x{{.+}} <col:25, line:[[@LINE-31]]:3> line:[[@LINE-32]]:31 class TestClassTemplate2 definition
  // CHECK-NEXT: | |-DefinitionData empty aggregate standard_layout trivially_copyable pod trivial literal has_constexpr_non_copy_move_ctor can_const_default_init
  // CHECK-NEXT: | | |-DefaultConstructor exists trivial constexpr needs_implicit defaulted_is_constexpr
  // CHECK-NEXT: | | |-CopyConstructor simple trivial has_const_param needs_implicit implicit_has_const_param
  // CHECK-NEXT: | | |-MoveConstructor exists simple trivial needs_implicit
  // CHECK-NEXT: | | |-CopyAssignment simple trivial has_const_param needs_implicit implicit_has_const_param
  // CHECK-NEXT: | | |-MoveAssignment exists simple trivial needs_implicit
  // CHECK-NEXT: | | `-Destructor simple irrelevant trivial needs_implicit
  // CHECK-NEXT: | `-CXXRecordDecl 0x{{.+}} <col:25, col:31> col:31 implicit class TestClassTemplate2
  // CHECK-NEXT: `-ClassTemplateSpecialization 0x{{.+}} 'TestClassTemplate2'

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

  // CHECK:      VarTemplateDecl 0x{{.+}} <{{.+}}:[[@LINE-11]]:7, col:43> col:43 TestVarTemplate
  // CHECK-NEXT: |-TemplateTypeParmDecl 0x{{.+}} <col:16, col:25> col:25 referenced typename depth 0 index 0 T
  // CHECK-NEXT: |-VarDecl 0x{{.+}} <col:28, col:43> col:43 TestVarTemplate 'const T' static
  // CHECK-NEXT: |-VarTemplateSpecializationDecl 0x{{.+}} parent 0x{{.+}} prev 0x{{.+}} <line:[[@LINE-11]]:3, col:34> col:14 referenced TestVarTemplate 'const int':'const int' cinit
  // CHECK-NEXT: | |-TemplateArgument type 'int'
  // CHECK-NEXT: | | `-BuiltinType 0x{{.+}} 'int'
  // CHECK-NEXT: | `-InitListExpr 0x{{.+}} <col:32, col:34> 'int':'int'
  // CHECK-NEXT: `-VarTemplateSpecializationDecl 0x{{.+}} <line:[[@LINE-18]]:28, col:43> col:43 referenced TestVarTemplate 'const int':'const int' static
  // CHECK-NEXT:   `-TemplateArgument type 'int'

  // CHECK:     VarTemplateSpecializationDecl 0x{{.+}} <{{.+}}:[[@LINE-21]]:28, col:43> col:43 referenced TestVarTemplate 'const int':'const int' static
  // CHECK-NEXT:`-TemplateArgument type 'int'
  // CHECK-NEXT:  `-BuiltinType 0x{{.+}} 'int'

  // CHECK:      VarTemplateDecl 0x{{.+}} parent 0x{{.+}} prev 0x{{.+}} <{{.+}}:[[@LINE-23]]:3, line:[[@LINE-22]]:34> col:14 TestVarTemplate
  // CHECK-NEXT: |-TemplateTypeParmDecl 0x{{.+}} <line:[[@LINE-24]]:12, col:21> col:21 referenced typename depth 0 index 0 T
  // CHECK-NEXT: |-VarDecl 0x{{.+}} parent 0x{{.+}} prev 0x{{.+}} <line:[[@LINE-24]]:3, col:34> col:14 TestVarTemplate 'const T' cinit
  // CHECK-NEXT: | `-InitListExpr 0x{{.+}} <col:32, col:34> 'void'
  // CHECK-NEXT: |-VarTemplateSpecialization 0x{{.+}} 'TestVarTemplate' 'const int':'const int'
  // CHECK-NEXT: `-VarTemplateSpecialization 0x{{.+}} 'TestVarTemplate' 'const int':'const int'
    
  // CHECK:      VarTemplateSpecializationDecl 0x{{.+}} parent 0x{{.+}} prev 0x{{.+}} <{{.+}}:[[@LINE-29]]:3, col:34> col:14 referenced TestVarTemplate 'const int':'const int' cinit
  // CHECK-NEXT: |-TemplateArgument type 'int'
  // CHECK-NEXT: | `-BuiltinType 0x{{.+}} 'int'
  // CHECK-NEXT: `-InitListExpr 0x{{.+}} <col:32, col:34> 'int':'int'
} 

template <class T>
class TestClassScopeFunctionSpecialization {
  template<class U> void foo(U a) { }
  template<> void foo<int>(int a) { }
};
// CHECK:      ClassScopeFunctionSpecializationDecl
// CHECK-NEXT:   CXXMethod{{.*}} foo 'void (int)'
// CHECK-NEXT:     ParmVarDecl
// CHECK-NEXT:     CompoundStmt
// CHECK-NEXT:   TemplateArgument{{.*}} 'int'

namespace TestTemplateTypeParmDecl {
  template<typename ... T, class U = int> void foo();
}
// CHECK:      NamespaceDecl{{.*}} TestTemplateTypeParmDecl
// CHECK-NEXT:   FunctionTemplateDecl
// CHECK-NEXT:     TemplateTypeParmDecl{{.*}} typename depth 0 index 0 ... T
// CHECK-NEXT:     TemplateTypeParmDecl{{.*}} class depth 0 index 1 U
// CHECK-NEXT:       TemplateArgument type 'int'

namespace TestNonTypeTemplateParmDecl {
  template<int I = 1, int ... J> void foo();
}
// CHECK:      NamespaceDecl{{.*}} TestNonTypeTemplateParmDecl
// CHECK-NEXT:   FunctionTemplateDecl
// CHECK-NEXT:     NonTypeTemplateParmDecl{{.*}} 'int' depth 0 index 0 I
// CHECK-NEXT:       TemplateArgument expr
// CHECK-NEXT:         IntegerLiteral{{.*}} 'int' 1
// CHECK-NEXT:     NonTypeTemplateParmDecl{{.*}} 'int' depth 0 index 1 ... J

namespace TestTemplateTemplateParmDecl {
  template<typename T> class A;
  template <template <typename> class T = A, template <typename> class ... U> void foo();
}
// CHECK:      NamespaceDecl{{.*}} TestTemplateTemplateParmDecl
// CHECK:        FunctionTemplateDecl
// CHECK-NEXT:     TemplateTemplateParmDecl{{.*}} T
// CHECK-NEXT:       TemplateTypeParmDecl{{.*}} typename
// CHECK-NEXT:       TemplateArgument{{.*}} template A
// CHECK-NEXT:     TemplateTemplateParmDecl{{.*}} ... U
// CHECK-NEXT:       TemplateTypeParmDecl{{.*}} typename

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
  // CHECK-NEXT:     Function{{.*}}foo

  template class testDecl<nullptr>;
  // CHECK:      ClassTemplateSpecializationDecl{{.*}} class testDecl
  // CHECK:        TemplateArgument{{.*}} nullptr

  template<int> class testIntegral { };
  template class testIntegral<1>;
  // CHECK:      ClassTemplateSpecializationDecl{{.*}} class testIntegral
  // CHECK:        TemplateArgument{{.*}} integral 1

  template<template<typename> class> class testTemplate { };
  template class testTemplate<A>;
  // CHECK:      ClassTemplateSpecializationDecl{{.*}} class testTemplate
  // CHECK:        TemplateArgument{{.*}} A

  template<template<typename> class ...T> class C {
    B<T...> testTemplateExpansion;
  };
  // FIXME: Need TemplateSpecializationType dumping to test TemplateExpansion.

  template<int, int = 0> class testExpr;
  template<int I> class testExpr<I> { };
  // CHECK:      ClassTemplatePartialSpecializationDecl{{.*}} class testExpr
  // CHECK:        TemplateArgument{{.*}} expr
  // CHECK-NEXT:     DeclRefExpr{{.*}}I

  template<int, int ...> class testPack { };
  template class testPack<0, 1, 2>;
  // CHECK:      ClassTemplateSpecializationDecl{{.*}} class testPack
  // CHECK:        TemplateArgument{{.*}} integral 0
  // CHECK-NEXT:   TemplateArgument{{.*}} pack
  // CHECK-NEXT:     TemplateArgument{{.*}} integral 1
  // CHECK-NEXT:     TemplateArgument{{.*}} integral 2
}

namespace testUsingDecl {
  int i;
}
namespace TestUsingDecl {
  using testUsingDecl::i;
}
// CHECK:      NamespaceDecl{{.*}} TestUsingDecl
// CHECK-NEXT:   UsingDecl{{.*}} testUsingDecl::i
// CHECK-NEXT:   UsingShadowDecl{{.*}} Var{{.*}} 'i' 'int'

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
// CHECK-NEXT:   LinkageSpecDecl{{.*}} C
// CHECK-NEXT:     FunctionDecl
// CHECK-NEXT:   LinkageSpecDecl{{.*}} C++
// CHECK-NEXT:     FunctionDecl

class TestAccessSpecDecl {
public:
private:
protected:
};
// CHECK:      CXXRecordDecl{{.*}} class TestAccessSpecDecl
// CHECK:         CXXRecordDecl{{.*}} class TestAccessSpecDecl
// CHECK-NEXT:    AccessSpecDecl{{.*}} public
// CHECK-NEXT:    AccessSpecDecl{{.*}} private
// CHECK-NEXT:    AccessSpecDecl{{.*}} protected

template<typename T> class TestFriendDecl {
  friend int foo();
  friend class A;
  friend T;
};
// CHECK:      CXXRecord{{.*}} TestFriendDecl
// CHECK:        CXXRecord{{.*}} TestFriendDecl
// CHECK-NEXT:   FriendDecl
// CHECK-NEXT:     FunctionDecl{{.*}} foo
// CHECK-NEXT:   FriendDecl{{.*}} 'class A':'A'
// CHECK-NEXT:     CXXRecordDecl{{.*}} class A
// CHECK-NEXT:   FriendDecl{{.*}} 'T'

namespace TestFileScopeAsmDecl {
  asm("ret");
}
// CHECK:      NamespaceDecl{{.*}} TestFileScopeAsmDecl{{$}}
// CHECK:        FileScopeAsmDecl{{.*> .*$}}
// CHECK-NEXT:     StringLiteral

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
