// RUN: %clang_cc1 -fsycl-is-device -std=c++17 -fsyntax-only -verify -DCPP17 %s
// RUN: %clang_cc1 -fsycl-is-device -std=c++20 -fsyntax-only -verify -DCPP20 %s

// Semantic tests for the sycl_external attribute.

// expected-error@+1{{'sycl_external' can only be applied to functions with external linkage}}
[[clang::sycl_external]]
static void func1() {}

// expected-error@+2{{'sycl_external' can only be applied to functions with external linkage}}
namespace {
  [[clang::sycl_external]]
  void func2() {}

  struct UnnX {};
}

// expected-error@+2{{'sycl_external' can only be applied to functions with external linkage}}
namespace { struct S4 {}; }
[[clang::sycl_external]] void func4(S4) {}

// FIXME: This case is currently being diagnosed as an error because clang implements
// default inheritance of attribute and explicit instantiation declaration names the
// symbol that causes the instantiated specialization to have internal linkage.
// expected-error@+3{{'sycl_external' can only be applied to functions with external linkage}}
namespace { struct S6 {}; }
template<typename>
[[clang::sycl_external]] void func6() {}
template void func6<S6>();
// expected-note@-1{{in instantiation of function template specialization 'func6<(anonymous namespace)::S6>' requested here}}

// expected-error@+3 2{{'sycl_external' can only be applied to functions with external linkage}}
namespace { struct S7 {}; }
template<typename>
[[clang::sycl_external]] void func7();
template<> void func7<S7>() {}
// expected-note@-1{{in instantiation of function template specialization 'func7<(anonymous namespace)::S7>' requested here}}

namespace { struct S8 {}; }
template<typename>
void func8();
template<> [[clang::sycl_external]] void func8<S8>() {}
// expected-error@-1{{'clang::sycl_external' attribute does not appear on the first declaration}}
// expected-note@-2{{previous declaration is here}}

// The first declaration of a SYCL external function is required to have this attribute.
// expected-note@+1{{previous declaration is here}}
int foo();
// expected-error@+1{{'clang::sycl_external' attribute does not appear on the first declaration}}
[[clang::sycl_external]] int foo();

// Subsequent declrations of a SYCL external function may optionally specify this attribute.
[[clang::sycl_external]] int boo();
[[clang::sycl_external]] int boo(); // OK
int boo(); // OK

class C {
  [[clang::sycl_external]] void member();
};

// expected-error@+1{{'sycl_external' cannot be applied to the 'main' function}}
[[clang::sycl_external]] int main()
{
  return 0;
}

// expected-error@+2{{'sycl_external' cannot be applied to an explicitly deleted function}}
class D {
  [[clang::sycl_external]] void del() = delete;
};
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

[[clang::sycl_external]] constexpr int square(int x);

// Devices that do not support the generic address space shall not specify
// a raw pointer or reference type as the return type or as a parameter type.
[[clang::sycl_external]] int *fun0();
[[clang::sycl_external]] int &fun1();
[[clang::sycl_external]] int &&fun2();
[[clang::sycl_external]] void fun3(int *);
[[clang::sycl_external]] void fun4(int &);
[[clang::sycl_external]] void fun5(int &&);

#if CPP20
[[clang::sycl_external]] consteval int func();
#endif

