// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux-gnu -fms-extensions -ast-dump -ast-dump-filter Test %s | FileCheck -check-prefix CHECK -strict-whitespace %s

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
// CHECK: VarDecl{{.*}} TestVarDeclNRVO 'class testVarDeclNRVO::A' nrvo

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
  class A { };
  class B { };
  class TestCXXRecordDecl : virtual A, public B {
    int i;
  };
}
// CHECK:      CXXRecordDecl{{.*}} class TestCXXRecordDecl
// CHECK-NEXT:   virtual private 'class testCXXRecordDecl::A'
// CHECK-NEXT:   public 'class testCXXRecordDecl::B'
// CHECK-NEXT:   CXXRecordDecl{{.*}} class TestCXXRecordDecl
// CHECK-NEXT:   FieldDecl

template<class...T>
class TestCXXRecordDeclPack : public T... {
};
// CHECK:      CXXRecordDecl{{.*}} class TestCXXRecordDeclPack
// CHECK-NEXT:   public 'T'...
// CHECK-NEXT:   CXXRecordDecl{{.*}} class TestCXXRecordDeclPack

thread_local int TestThreadLocalInt;
// CHECK: TestThreadLocalInt {{.*}} tls_dynamic

__module_private__ class TestCXXRecordDeclPrivate;
// CHECK: CXXRecordDecl{{.*}} class TestCXXRecordDeclPrivate __module_private__

class testCXXMethodDecl {
  __module_private__ void TestCXXMethodDeclPrivate();
  virtual void TestCXXMethodDeclPure() = 0;
  void TestCXXMethodDeclDelete() = delete;
  void TestCXXMethodDeclThrow() throw();
  void TestCXXMethodDeclThrowType() throw(int);
};
// CHECK: CXXMethodDecl{{.*}} TestCXXMethodDeclPrivate 'void (void)' __module_private__
// CHECK: CXXMethodDecl{{.*}} TestCXXMethodDeclPure 'void (void)' virtual pure
// CHECK: CXXMethodDecl{{.*}} TestCXXMethodDeclDelete 'void (void)' delete
// CHECK: CXXMethodDecl{{.*}} TestCXXMethodDeclThrow 'void (void) throw()'
// CHECK: CXXMethodDecl{{.*}} TestCXXMethodDeclThrowType 'void (void) throw(int)'

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
// CHECK:      CXXDestructorDecl{{.*}} ~TestCXXDestructorDecl 'void (void) noexcept'
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
// CHECK:      CXXConversionDecl{{.*}} operator int 'int (void)'
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
// CHECK:      FunctionTemplateDecl{{.*}} TestFunctionTemplate
// CHECK-NEXT:   TemplateTypeParmDecl
// CHECK-NEXT:   FunctionDecl{{.*}} TestFunctionTemplate 'void (T)'
// CHECK-NEXT:     ParmVarDecl{{.*}} 'T'
// CHECK-NEXT:     CompoundStmt
// CHECK-NEXT:   FunctionDecl{{.*}} TestFunctionTemplate {{.*}}A
// CHECK-NEXT:     TemplateArgument
// CHECK-NEXT:     ParmVarDecl
// CHECK-NEXT:     CompoundStmt
// CHECK-NEXT:   Function{{.*}} 'TestFunctionTemplate' {{.*}}B
// CHECK-NEXT:   FunctionDecl{{.*}} TestFunctionTemplate {{.*}}C
// CHECK-NEXT:     TemplateArgument
// CHECK-NEXT:     ParmVarDecl
// CHECK-NEXT:   FunctionDecl{{.*}} TestFunctionTemplate {{.*}}D
// CHECK-NEXT:     TemplateArgument
// CHECK-NEXT:     ParmVarDecl
// CHECK-NEXT:     CompoundStmt
// CHECK:      FunctionDecl{{.*}} TestFunctionTemplate {{.*}}B
// CHECK-NEXT:   TemplateArgument
// CHECK-NEXT:   ParmVarDecl

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
}
// CHECK:      ClassTemplateDecl{{.*}} TestClassTemplate
// CHECK-NEXT:   TemplateTypeParmDecl
// CHECK-NEXT:   CXXRecordDecl{{.*}} class TestClassTemplate
// CHECK-NEXT:     CXXRecordDecl{{.*}} class TestClassTemplate
// CHECK-NEXT:     AccessSpecDecl{{.*}} public
// CHECK-NEXT:     CXXConstructorDecl{{.*}} <line:{{.*}}:5, col:23>
// CHECK-NEXT:     CXXDestructorDecl{{.*}} <line:{{.*}}:5, col:24>
// CHECK-NEXT:     CXXMethodDecl{{.*}} <line:{{.*}}:5, col:11>
// CHECK-NEXT:     FieldDecl{{.*}} i
// CHECK-NEXT:   ClassTemplateSpecializationDecl{{.*}} class TestClassTemplate
// CHECK-NEXT:     TemplateArgument{{.*}}A
// CHECK-NEXT:     CXXRecordDecl{{.*}} class TestClassTemplate
// CHECK-NEXT:     AccessSpecDecl{{.*}} public
// CHECK-NEXT:     CXXConstructorDecl{{.*}} <line:{{.*}}:5, col:23>
// CHECK-NEXT:     CXXDestructorDecl{{.*}} <line:{{.*}}:5, col:24>
// CHECK-NEXT:     CXXMethodDecl{{.*}} <line:{{.*}}:5, col:11>
// CHECK-NEXT:     FieldDecl{{.*}} i
// CHECK:        ClassTemplateSpecialization{{.*}} 'TestClassTemplate'
// CHECK-NEXT:   ClassTemplateSpecialization{{.*}} 'TestClassTemplate'
// CHECK-NEXT:   ClassTemplateSpecialization{{.*}} 'TestClassTemplate'

// CHECK:      ClassTemplateSpecializationDecl{{.*}} class TestClassTemplate
// CHECK-NEXT:   TemplateArgument{{.*}}B
// CHECK-NEXT:   CXXRecordDecl{{.*}} class TestClassTemplate
// CHECK-NEXT:   FieldDecl{{.*}} j

// CHECK:      ClassTemplateSpecializationDecl{{.*}} class TestClassTemplate
// CHECK-NEXT:   TemplateArgument{{.*}}C
// CHECK-NEXT:   CXXRecordDecl{{.*}} class TestClassTemplate
// CHECK-NEXT:   AccessSpecDecl{{.*}} public
// CHECK-NEXT:   CXXConstructorDecl{{.*}} <line:{{.*}}:5, col:23>
// CHECK-NEXT:   CXXDestructorDecl{{.*}} <line:{{.*}}:5, col:24>
// CHECK-NEXT:   CXXMethodDecl{{.*}} <line:{{.*}}:5, col:11>
// CHECK-NEXT:   FieldDecl{{.*}} i

// CHECK:      ClassTemplateSpecializationDecl{{.*}} class TestClassTemplate
// CHECK-NEXT:   TemplateArgument{{.*}}D
// CHECK-NEXT:   CXXRecordDecl{{.*}} class TestClassTemplate
// CHECK-NEXT:   AccessSpecDecl{{.*}} public
// CHECK-NEXT:   CXXConstructorDecl{{.*}} <line:{{.*}}:5, col:23>
// CHECK-NEXT:   CXXDestructorDecl{{.*}} <line:{{.*}}:5, col:24>
// CHECK-NEXT:   CXXMethodDecl{{.*}} <line:{{.*}}:5, col:11>
// CHECK-NEXT:   FieldDecl{{.*}} i

// CHECK:      ClassTemplatePartialSpecializationDecl{{.*}} class TestClassTemplatePartial
// CHECK-NEXT:   TemplateArgument
// CHECK-NEXT:   TemplateArgument{{.*}}A
// CHECK-NEXT:   TemplateTypeParmDecl
// CHECK-NEXT:   CXXRecordDecl{{.*}} class TestClassTemplatePartial
// CHECK-NEXT:   FieldDecl{{.*}} j

// PR15220 dump instantiation only once
namespace testCanonicalTemplate {
  class A {};

  template<typename T> void TestFunctionTemplate(T);
  template<typename T> void TestFunctionTemplate(T);
  void bar(A a) { TestFunctionTemplate(a); }
  // CHECK:      FunctionTemplateDecl{{.*}} TestFunctionTemplate
  // CHECK-NEXT:   TemplateTypeParmDecl
  // CHECK-NEXT:   FunctionDecl{{.*}} TestFunctionTemplate 'void (T)'
  // CHECK-NEXT:     ParmVarDecl{{.*}} 'T'
  // CHECK-NEXT:   FunctionDecl{{.*}} TestFunctionTemplate {{.*}}A
  // CHECK-NEXT:     TemplateArgument
  // CHECK-NEXT:     ParmVarDecl
  // CHECK:      FunctionTemplateDecl{{.*}} TestFunctionTemplate
  // CHECK-NEXT:   TemplateTypeParmDecl
  // CHECK-NEXT:   FunctionDecl{{.*}} TestFunctionTemplate 'void (T)'
  // CHECK-NEXT:     ParmVarDecl{{.*}} 'T'
  // CHECK-NEXT:   Function{{.*}} 'TestFunctionTemplate'
  // CHECK-NOT:      TemplateArgument

  template<typename T1> class TestClassTemplate {
    template<typename T2> friend class TestClassTemplate;
  };
  TestClassTemplate<A> a;
  // CHECK:      ClassTemplateDecl{{.*}} TestClassTemplate
  // CHECK-NEXT:   TemplateTypeParmDecl
  // CHECK-NEXT:   CXXRecordDecl{{.*}} class TestClassTemplate
  // CHECK-NEXT:     CXXRecordDecl{{.*}} class TestClassTemplate
  // CHECK-NEXT:     FriendDecl
  // CHECK-NEXT:       ClassTemplateDecl{{.*}} TestClassTemplate
  // CHECK-NEXT:         TemplateTypeParmDecl
  // CHECK-NEXT:         CXXRecordDecl{{.*}} class TestClassTemplate
  // CHECK-NEXT:   ClassTemplateSpecializationDecl{{.*}} class TestClassTemplate
  // CHECK-NEXT:     TemplateArgument{{.*}}A
  // CHECK-NEXT:     CXXRecordDecl{{.*}} class TestClassTemplate
}

template <class T>
class TestClassScopeFunctionSpecialization {
  template<class U> void foo(U a) { }
  template<> void foo<int>(int a) { }
};
// CHECK:      ClassScopeFunctionSpecializationDecl
// CHECK-NEXT:   CXXMethod{{.*}} 'foo' 'void (int)'
// CHECK-NEXT:   TemplateArgument{{.*}} 'int'

namespace TestTemplateTypeParmDecl {
  template<typename ... T, class U = int> void foo();
}
// CHECK:      NamespaceDecl{{.*}} TestTemplateTypeParmDecl
// CHECK-NEXT:   FunctionTemplateDecl
// CHECK-NEXT:     TemplateTypeParmDecl{{.*}} typename ... T
// CHECK-NEXT:     TemplateTypeParmDecl{{.*}} class U
// CHECK-NEXT:       TemplateArgument type 'int'

namespace TestNonTypeTemplateParmDecl {
  template<int I = 1, int ... J> void foo();
}
// CHECK:      NamespaceDecl{{.*}} TestNonTypeTemplateParmDecl
// CHECK-NEXT:   FunctionTemplateDecl
// CHECK-NEXT:     NonTypeTemplateParmDecl{{.*}} 'int' I
// CHECK-NEXT:       TemplateArgument expr
// CHECK-NEXT:         IntegerLiteral{{.*}} 'int' 1
// CHECK-NEXT:     NonTypeTemplateParmDecl{{.*}} 'int' ... J

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
  // CHECK-NEXT:   TemplateArgument{{.*}} type 'int'

  template<int fp(void)> class testDecl { };
  template class testDecl<foo>;
  // CHECK:      ClassTemplateSpecializationDecl{{.*}} class testDecl
  // CHECK-NEXT:   TemplateArgument{{.*}} decl
  // CHECK-NEXT:     Function{{.*}}foo

  template class testDecl<nullptr>;
  // CHECK:      ClassTemplateSpecializationDecl{{.*}} class testDecl
  // CHECK-NEXT:   TemplateArgument{{.*}} nullptr

  template<int> class testIntegral { };
  template class testIntegral<1>;
  // CHECK:      ClassTemplateSpecializationDecl{{.*}} class testIntegral
  // CHECK-NEXT:   TemplateArgument{{.*}} integral 1

  template<template<typename> class> class testTemplate { };
  template class testTemplate<A>;
  // CHECK:      ClassTemplateSpecializationDecl{{.*}} class testTemplate
  // CHECK-NEXT:   TemplateArgument{{.*}} A

  template<template<typename> class ...T> class C {
    B<T...> testTemplateExpansion;
  };
  // FIXME: Need TemplateSpecializationType dumping to test TemplateExpansion.

  template<int, int = 0> class testExpr;
  template<int I> class testExpr<I> { };
  // CHECK:      ClassTemplatePartialSpecializationDecl{{.*}} class testExpr
  // CHECK-NEXT:   TemplateArgument{{.*}} expr
  // CHECK-NEXT:     DeclRefExpr{{.*}}I

  template<int, int ...> class testPack { };
  template class testPack<0, 1, 2>;
  // CHECK:      ClassTemplateSpecializationDecl{{.*}} class testPack
  // CHECK-NEXT:   TemplateArgument{{.*}} integral 0
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
// CHECK-NEXT:    CXXRecordDecl{{.*}} class TestAccessSpecDecl
// CHECK-NEXT:    AccessSpecDecl{{.*}} public
// CHECK-NEXT:    AccessSpecDecl{{.*}} private
// CHECK-NEXT:    AccessSpecDecl{{.*}} protected

template<typename T> class TestFriendDecl {
  friend int foo();
  friend class A;
  friend T;
};
// CHECK:      CXXRecord{{.*}} TestFriendDecl
// CHECK-NEXT:   CXXRecord{{.*}} TestFriendDecl
// CHECK-NEXT:   FriendDecl
// CHECK-NEXT:     FunctionDecl{{.*}} foo
// CHECK-NEXT:   FriendDecl{{.*}} 'class A':'class A'
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
// CHECK: |-FunctionDecl [[TestFriendDecl2_f:0x.*]] <{{.*}}> {{.*}} f 'void (void)'
// CHECK: `-CXXRecordDecl {{.*}} struct S
// CHECK:   |-CXXRecordDecl {{.*}} struct S
// CHECK:   `-FriendDecl
// CHECK:     `-FunctionDecl {{.*}} parent [[TestFriendDecl2]] prev [[TestFriendDecl2_f]] <{{.*}}> {{.*}} f 'void (void)'

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
