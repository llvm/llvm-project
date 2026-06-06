// RUN: %clang_cc1 -fsycl-is-host -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsycl-is-host -std=c++20 -fsyntax-only -verify -DCPP20 %s
// RUN: %clang_cc1 -fsycl-is-device -std=c++20 -fsyntax-only -verify -DCPP20 %s

// Semantic tests for the sycl_external attribute.

// expected-error@+1{{'clang::sycl_external' can only be applied to functions with external linkage}}
[[clang::sycl_external]]
static void func1() {}

// expected-error@+2{{'clang::sycl_external' can only be applied to functions with external linkage}}
namespace {
  [[clang::sycl_external]]
  void func2() {}
}

// expected-error@+2{{'clang::sycl_external' can only be applied to functions with external linkage}}
namespace { struct S4 {}; }
[[clang::sycl_external]] void func4(S4) {}

// expected-error@+3{{'clang::sycl_external' can only be applied to functions with external linkage}}
namespace { struct S5 {}; }
template<typename> [[clang::sycl_external]] void func5();
template<> [[clang::sycl_external]] void func5<S5>() {}

namespace { struct S6 {}; }
template<typename>
[[clang::sycl_external]] void func6() {}
template void func6<S6>();

// FIXME: C++23 [temp.expl.spec]p12 states:
//   ... Similarly, attributes appearing in the declaration of a template
//   have no effect on an explicit specialization of that template.
// Clang currently instantiates and propagates attributes from a function
// template to its explicit specializations resulting in the following
// spurious error.
// expected-error@+3{{'clang::sycl_external' can only be applied to functions with external linkage}}
namespace { struct S7 {}; }
template<typename>
[[clang::sycl_external]] void func7();
template<> void func7<S7>() {}

// FIXME: The explicit function template specialization appears to trigger
// instantiation of a declaration from the primary template without the
// attribute leading to a spurious diagnostic that the sycl_external
// attribute is not present on the first declaration.
namespace { struct S8 {}; }
template<typename>
void func8();
template<> [[clang::sycl_external]] void func8<S8>() {}
// expected-warning@-1{{'clang::sycl_external' attribute does not appear on the first declaration}}
// expected-error@-2{{'clang::sycl_external' can only be applied to functions with external linkage}}
// expected-note@-3{{previous declaration is here}}

namespace { struct S9 {}; }
struct T9 {
  using type = S9;
};
template<typename>
[[clang::sycl_external]] void func9() {}
template<typename T>
[[clang::sycl_external]] void test_func9() {
  func9<typename T::type>();
}
template void test_func9<T9>();

// The first declaration of a SYCL external function is required to have this attribute.
// expected-note@+1{{previous declaration is here}}
int foo();
// expected-warning@+1{{'clang::sycl_external' attribute does not appear on the first declaration}}
[[clang::sycl_external]] int foo();

// expected-note@+1{{previous declaration is here}}
void goo();
// expected-warning@+1{{'clang::sycl_external' attribute does not appear on the first declaration}}
[[clang::sycl_external]] void goo();
void goo() {}

// expected-note@+1{{previous declaration is here}}
void hoo() {}
// expected-warning@+1{{'clang::sycl_external' attribute does not appear on the first declaration}}
[[clang::sycl_external]] void hoo();

// expected-note@+1{{previous declaration is here}}
void joo();
void use_joo() {
  joo();
}
// expected-warning@+1{{'clang::sycl_external' attribute does not appear on the first declaration}}
[[clang::sycl_external]] void joo();

// Subsequent declarations of a SYCL external function may optionally specify this attribute.
[[clang::sycl_external]] int boo();
[[clang::sycl_external]] int boo(); // OK
int boo(); // OK

class C {
  [[clang::sycl_external]] void member();
};

// expected-error@+1{{'clang::sycl_external' cannot be applied to the 'main' function}}
[[clang::sycl_external]] int main()
{
  return 0;
}

// expected-error@+2{{'clang::sycl_external' cannot be applied to an explicitly deleted function}}
class D {
  [[clang::sycl_external]] void mdel() = delete;
};

// expected-error@+1{{'clang::sycl_external' cannot be applied to an explicitly deleted function}}
[[clang::sycl_external]] void del() = delete;

struct NonCopyable {
  ~NonCopyable() = delete;
  [[clang::sycl_external]] NonCopyable(const NonCopyable&) = default;
};

class A {
  [[clang::sycl_external]]
  A() {}

  [[clang::sycl_external]] void mf() {}
  [[clang::sycl_external]] static void smf();
};

class B {
public:
  [[clang::sycl_external]] virtual void foo() {}

  [[clang::sycl_external]] virtual void bar() = 0;
};
[[clang::sycl_external]] void B::bar() {}

[[clang::sycl_external]] constexpr int square(int x);

// Devices that do not support the generic address space shall not specify
// a raw pointer or reference type as the return type or as a parameter type.
[[clang::sycl_external]] int *fun0();
[[clang::sycl_external]] int &fun1();
[[clang::sycl_external]] int &&fun2();
[[clang::sycl_external]] void fun3(int *);
[[clang::sycl_external]] void fun4(int &);
[[clang::sycl_external]] void fun5(int &&);
template<typename T>
[[clang::sycl_external]] void fun6(T) {}
template void fun6(int *);
template<> [[clang::sycl_external]] void fun6<long*>(long *) {}

#if CPP20
[[clang::sycl_external]] consteval int func();
#endif
