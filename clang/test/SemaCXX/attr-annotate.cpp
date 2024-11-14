// RUN: split-file %s %t
// RUN: %clang_cc1 -std=gnu++20 -fsyntax-only -ast-dump -verify %t/good_annotate.cpp | FileCheck %s
// RUN: %clang_cc1 -std=gnu++20 -fsyntax-only -verify %t/bad_annotate.cpp
//--- good_annotate.cpp
// expected-no-diagnostics

void f() {
  [[clang::annotate("decl", 1)]] int i = 0;
  [[clang::annotate("stmt", 2)]] i += 1;
[[clang::annotate("label", 3)]] label1:
  i += 2;
}

// CHECK: -FunctionDecl 0x{{[0-9a-z]+}} {{.*:[0-9]+:[0-9]+, line:[0-9]+:[0-9]+> line:[0-9]+:[0-9]+}} f 'void ()'
// CHECK: -VarDecl 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?> col:[0-9]+}} used i 'int'
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} "decl"
// CHECK: -IntegerLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'int' 1
// CHECK: -AttributedStmt 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}}
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} "stmt"
// CHECK: -IntegerLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'int' 2
// CHECK: -LabelStmt 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, line:[0-9]+:[0-9]+)?>}} 'label1'
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}} "label"
// CHECK: -IntegerLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'int' 3
// CHECK: -CompoundAssignOperator 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}}

template <typename T> void g() {
  [[clang::annotate("tmpl_decl", 4)]] T j = 0;
  [[clang::annotate("tmpl_stmt", 5)]] j += 1;
[[clang::annotate("tmpl_label", 6)]] label2:
  j += 2;
}

// CHECK: -FunctionTemplateDecl 0x{{[0-9a-z]+}} {{.*:[0-9]+:[0-9]+, line:[0-9]+:[0-9]+> line:[0-9]+:[0-9]+}} g
// CHECK: -VarDecl 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?> col:[0-9]+}} referenced j 'T'
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} "tmpl_decl"
// CHECK: -IntegerLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'int' 4
// CHECK: -AttributedStmt 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}}
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} "tmpl_stmt"
// CHECK: -IntegerLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'int' 5
// CHECK: -LabelStmt 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, line:[0-9]+:[0-9]+)?>}} 'label2'
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}} "tmpl_label"
// CHECK: -IntegerLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'int' 6
// CHECK: -CompoundAssignOperator 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}}

void h() {
  g<int>();
}

// CHECK: -FunctionDecl 0x{{[0-9a-z]+}} {{.*:[0-9]+:[0-9]+, line:[0-9]+:[0-9]+> line:[0-9]+:[0-9]+}} used g 'void ()' implicit_instantiation
// CHECK: -VarDecl 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?> col:[0-9]+}} used j 'int'
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} "tmpl_decl"
// CHECK: -IntegerLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'int' 4
// CHECK: -AttributedStmt 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}}
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} Implicit "tmpl_stmt"
// CHECK: -IntegerLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'int' 5
// CHECK: -LabelStmt 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, line:[0-9]+:[0-9]+)?>}} 'label2'
// CHECK: -AnnotateAttr 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}} "tmpl_label"
// CHECK: -IntegerLiteral 0x{{[0-9a-z]+}} {{<col:[0-9]+(, col:[0-9]+)?>}} 'int' 6
// CHECK: -CompoundAssignOperator 0x{{[0-9a-z]+}} {{<line:[0-9]+:[0-9]+(, col:[0-9]+)?>}}

//--- bad_annotate.cpp

template<bool If, typename Type>
struct enable_if {
  using type= Type;
};

template<typename Type>
struct enable_if<false, Type> {};

template<typename T1, typename T2>
struct is_same {
  static constexpr bool value = false;
};

template<typename T1>
struct is_same<T1, T1> {
  static constexpr bool value = true;
};

constexpr const char *str() {
  return "abc";
}

template<typename T, typename enable_if<!is_same<int, T>::value, int>::type = 0>
constexpr T fail_on_int(T t) {return t;}
// expected-note@-1 {{candidate template ignored: requirement}}

namespace test0 {
  template<typename T, T v>
  struct A {
    [[clang::annotate("test", fail_on_int(v))]] void t() {}
    // expected-error@-1 {{no matching function for call to 'fail_on_int'}}
    [[clang::annotate("test", (typename enable_if<!is_same<long, T>::value, int>::type)v)]] void t1() {}
    // expected-error@-1 {{failed requirement}}
  };
  A<int, 9> a;
// expected-note@-1 {{in instantiation of template class}}
  A<long, 7> a1;
// expected-note@-1 {{in instantiation of template class}}
  A<unsigned long, 6> a2;

  template<typename T>
  struct B {
    [[clang::annotate("test", ((void)T{}, 9))]] void t() {}
    // expected-error@-1 {{cannot create object of function type 'void ()'}}
  };
  B<int> b;
  B<void ()> b1;
// expected-note@-1 {{in instantiation of template class}}
}

namespace test1 {
int g_i; // expected-note {{declared here}}

[[clang::annotate("test", "arg")]] void t3() {}

template <typename T, T V>
struct B {
  static T b; // expected-note {{declared here}}
  static constexpr T cb = V;
  template <typename T1, T1 V1>
  struct foo {
    static T1 f; // expected-note {{declared here}}
    static constexpr T1 cf = V1;
    int v __attribute__((annotate("v_ann_0", str(), 90, V, g_i))) __attribute__((annotate("v_ann_1", V1)));
    // expected-error@-1 {{'annotate' attribute requires parameter 4 to be a constant expression}}
    // expected-note@-2 {{is not allowed in a constant expression}}
    [[clang::annotate("qdwqwd", cf, cb)]] void t() {}
    [[clang::annotate("qdwqwd", f, cb)]] void t1() {}
    // expected-error@-1 {{'annotate' attribute requires parameter 1 to be a constant expression}}
    // expected-note@-2 {{is not allowed in a constant expression}}
    [[clang::annotate("jui", b, cf)]] void t2() {}
    // expected-error@-1 {{'annotate' attribute requires parameter 1 to be a constant expression}}
    // expected-note@-2 {{is not allowed in a constant expression}}
    [[clang::annotate("jui", ((void)b, 0), cf)]] [[clang::annotate("jui", &b, cf, &foo::t2, str())]] void t3() {}
  };
};

static B<int long, -1>::foo<unsigned, 9> gf; // expected-note {{in instantiation of}}
static B<int long, -2> gf1;

} // namespace test1

namespace test2 {

template<int I>
int f() {
  [[clang::annotate("test", I)]] int v = 0; // expected-note {{declared here}}
  [[clang::annotate("test", v)]] int v2 = 0;
  // expected-error@-1 {{'annotate' attribute requires parameter 1 to be a constant expression}}
  // expected-note@-2 {{is not allowed in a constant expression}}
  [[clang::annotate("test", rtyui)]] int v3 = 0;
    // expected-error@-1 {{use of undeclared identifier 'rtyui'}}
}

void test() {}
}

namespace test3 {

void f() {
  int n = 10;
  int vla[n];

  [[clang::annotate("vlas are awful", sizeof(vla))]] int i = 0; // reject, the sizeof is not unevaluated
  // expected-error@-1 {{'annotate' attribute requires parameter 1 to be a constant expression}}
  // expected-note@-2 {{subexpression not valid in a constant expression}}
  [[clang::annotate("_Generic selection expression should be fine", _Generic(n, int : 0, default : 1))]]
  int j = 0; // second arg should resolve to 0 fine
}
void designator();
[[clang::annotate("function designators?", designator)]] int k = 0; // Should work?

void self() {
  [[clang::annotate("function designators?", self)]] int k = 0;
}

}

namespace test4 {
constexpr int foldable_but_invalid() {
  int *A = new int(0);
// expected-note@-1 {{allocation performed here was not deallocated}}
  return *A;
}

[[clang::annotate("", foldable_but_invalid())]] void f1() {}
// expected-error@-1 {{'annotate' attribute requires parameter 1 to be a constant expression}}

[[clang::annotate()]] void f2() {}
// expected-error@-1 {{'annotate' attribute takes at least 1 argument}}

template <typename T> [[clang::annotate()]] void f2() {}
// expected-error@-1 {{'annotate' attribute takes at least 1 argument}}
}
