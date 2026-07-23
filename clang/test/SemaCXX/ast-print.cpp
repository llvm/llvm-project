// RUN: %clang_cc1 -triple %ms_abi_triple -ast-print %s -std=gnu++11 | FileCheck %s

// CHECK: r;
// CHECK-NEXT: (r->method());
struct MyClass
{
    void method() {}
};

struct Reference
{
    MyClass* object;
    MyClass* operator ->() { return object; }
};

void test1() {
    Reference r;
    (r->method());
}

// CHECK: if (int a = 1)
// CHECK:  while (int a = 1)
// CHECK:  switch (int a = 1)
// CHECK:  for (; int a = 1;)

void test2()
{
    if (int a = 1) { }
    while (int a = 1) { }
    switch (int a = 1) { }
    for(; int a = 1; ) { }
}

// CHECK: new (1) int;
void *operator new (typeof(sizeof(1)), int, int = 2);
void test3() {
  new (1) int;
}

// CHECK: new X;
struct X {
  void *operator new (typeof(sizeof(1)), int = 2);
};
void test4() { new X; }

// CHECK: for (int i = 2097, j = 42; false;)
void test5() {
  for (int i = 2097, j = 42; false;) {}
}

// CHECK: test6fn((int &)y);
void test6fn(int& x);
void test6() {
    unsigned int y = 0;
    test6fn((int&)y);
}

// CHECK: S s(1, 2);

template <class S> void test7()
{
    S s( 1,2 );
}


// CHECK: t.~T();

template <typename T> void test8(T t) { t.~T(); }


// CHECK:      enum E
// CHECK-NEXT:  A,
// CHECK-NEXT:  B,
// CHECK-NEXT:  C
// CHECK-NEXT:  };
// CHECK-NEXT: {{^[ ]+}}E a = A;

struct test9
{
    void f()
    {
        enum E { A, B, C };
        E a = A;
    }
};

namespace test10 {
  namespace M {
    template<typename T>
    struct X {
      enum { value };
    };
  }
}

typedef int INT;

// CHECK: test11
// CHECK-NEXT: return test10::M::X<INT>::value;
int test11() {
  return test10::M::X<INT>::value;
}


struct DefaultArgClass
{
  DefaultArgClass(int a = 1) {}
  DefaultArgClass(int a, int b, int c = 1) {}
};

struct NoArgClass
{
  NoArgClass() {}
};

struct VirualDestrClass
{
  VirualDestrClass(int arg);
  virtual ~VirualDestrClass();
};

struct ConstrWithCleanupsClass
{
  ConstrWithCleanupsClass(const VirualDestrClass& cplx = VirualDestrClass(42));
};

// CHECK: test12
// CHECK-NEXT: DefaultArgClass useDefaultArg;
// CHECK-NEXT: DefaultArgClass overrideDefaultArg(1);
// CHECK-NEXT: DefaultArgClass(1, 2);
// CHECK-NEXT: DefaultArgClass(1, 2, 3);
// CHECK-NEXT: NoArgClass noArg;
// CHECK-NEXT: ConstrWithCleanupsClass cwcNoArg;
// CHECK-NEXT: ConstrWithCleanupsClass cwcOverrideArg(48);
// CHECK-NEXT: ConstrWithCleanupsClass cwcExplicitArg(VirualDestrClass(56));
void test12() {
  DefaultArgClass useDefaultArg;
  DefaultArgClass overrideDefaultArg(1);
  DefaultArgClass tempWithDefaultArg = DefaultArgClass(1, 2);
  DefaultArgClass tempWithExplictArg = DefaultArgClass(1, 2, 3);
  NoArgClass noArg;
  ConstrWithCleanupsClass cwcNoArg;
  ConstrWithCleanupsClass cwcOverrideArg(48);
  ConstrWithCleanupsClass cwcExplicitArg(VirualDestrClass(56));
}

// CHECK: void test13() {
// CHECK:   _Atomic(int) i;
// CHECK:   __c11_atomic_init(&i, 0);
// CHECK:   __c11_atomic_load(&i, 0);
// CHECK: }
void test13() {
  _Atomic(int) i;
  __c11_atomic_init(&i, 0);
  __c11_atomic_load(&i, 0);
}


// CHECK: void test14() {
// CHECK:     struct X {
// CHECK:         union {
// CHECK:             int x;
// CHECK:         } x;
// CHECK:     };
// CHECK: }
void test14() {
  struct X { union { int x; } x; };
}


// CHECK: float test15() {
// CHECK:     return __builtin_asinf(1.F);
// CHECK: }
// CHECK-NOT: extern "C"
float test15() {
  return __builtin_asinf(1.0F);
}

// CHECK: void test_atomic_loads(int *ptr, int *ret, int memorder) {
// CHECK:   __atomic_load_n(ptr, memorder);
// CHECK:   __atomic_load(ptr, ret, memorder);
// CHECK: }
void test_atomic_loads(int *ptr, int *ret, int memorder) {
  __atomic_load_n(ptr, memorder);
  __atomic_load(ptr, ret, memorder);
}

// CHECK: void test_atomic_stores(int *ptr, int val, int memorder) {
// CHECK:   __atomic_store_n(ptr, val, memorder);
// CHECK:   __atomic_store(ptr, &val, memorder);
// CHECK: }
void test_atomic_stores(int *ptr, int val, int memorder) {
  __atomic_store_n(ptr, val, memorder);
  __atomic_store(ptr, &val, memorder);
}

// CHECK: void test_atomic_exchanges(int *ptr, int val, int *ret, int memorder) {
// CHECK:   __atomic_exchange_n(ptr, val, memorder);
// CHECK:   __atomic_exchange(ptr, &val, ret, memorder);
// CHECK: }
void test_atomic_exchanges(int *ptr, int val, int *ret, int memorder) {
  __atomic_exchange_n(ptr, val, memorder);
  __atomic_exchange(ptr, &val, ret, memorder);
}

// CHECK: void test_atomic_cmpxchgs(int *ptr, int *expected, int desired, bool weak, int success_memorder, int failure_memorder) {
// CHECK:   __atomic_compare_exchange_n(ptr, expected, desired, weak, success_memorder, failure_memorder);
// CHECK:   __atomic_compare_exchange(ptr, expected, &desired, weak, success_memorder, failure_memorder);
// CHECK: }
void test_atomic_cmpxchgs(int *ptr, int *expected, int desired, bool weak, int success_memorder, int failure_memorder) {
  __atomic_compare_exchange_n(ptr, expected, desired, weak, success_memorder, failure_memorder);
  __atomic_compare_exchange(ptr, expected, &desired, weak, success_memorder, failure_memorder);
}

// CHECK: void test_atomic_fetch_ops(int *ptr, int val, int memorder) {
// CHECK:   __atomic_add_fetch(ptr, val, memorder);
// CHECK:   __atomic_sub_fetch(ptr, val, memorder);
// CHECK:   __atomic_and_fetch(ptr, val, memorder);
// CHECK:   __atomic_xor_fetch(ptr, val, memorder);
// CHECK:   __atomic_or_fetch(ptr, val, memorder);
// CHECK:   __atomic_nand_fetch(ptr, val, memorder);
// CHECK:   __atomic_fetch_add(ptr, val, memorder);
// CHECK:   __atomic_fetch_sub(ptr, val, memorder);
// CHECK:   __atomic_fetch_and(ptr, val, memorder);
// CHECK:   __atomic_fetch_xor(ptr, val, memorder);
// CHECK:   __atomic_fetch_or(ptr, val, memorder);
// CHECK:   __atomic_fetch_nand(ptr, val, memorder);
// CHECK: }
void test_atomic_fetch_ops(int *ptr, int val, int memorder) {
  __atomic_add_fetch(ptr, val, memorder);
  __atomic_sub_fetch(ptr, val, memorder);
  __atomic_and_fetch(ptr, val, memorder);
  __atomic_xor_fetch(ptr, val, memorder);
  __atomic_or_fetch(ptr, val, memorder);
  __atomic_nand_fetch(ptr, val, memorder);
  __atomic_fetch_add(ptr, val, memorder);
  __atomic_fetch_sub(ptr, val, memorder);
  __atomic_fetch_and(ptr, val, memorder);
  __atomic_fetch_xor(ptr, val, memorder);
  __atomic_fetch_or(ptr, val, memorder);
  __atomic_fetch_nand(ptr, val, memorder);
}

// CHECK: void test_atomic_setclear(void *ptr, int memorder) {
// CHECK:   __atomic_test_and_set(ptr, memorder);
// CHECK:   __atomic_clear(ptr, memorder);
// CHECK: }
void test_atomic_setclear(void *ptr, int memorder) {
  __atomic_test_and_set(ptr, memorder);
  __atomic_clear(ptr, memorder);
}

// CHECK: void test_atomic_fences(int memorder) {
// CHECK:   __atomic_thread_fence(memorder);
// CHECK:   __atomic_signal_fence(memorder);
// CHECK: }
void test_atomic_fences(int memorder) {
  __atomic_thread_fence(memorder);
  __atomic_signal_fence(memorder);
}

// CHECK: void test_atomic_lockfree(unsigned long size, void *ptr) {
// CHECK:   __atomic_always_lock_free(size, ptr);
// CHECK:   __atomic_is_lock_free(size, ptr);
// CHECK: }
void test_atomic_lockfree(unsigned long size, void *ptr) {
  __atomic_always_lock_free(size, ptr);
  __atomic_is_lock_free(size, ptr);
}

namespace PR18776 {
struct A {
  operator void *();
  explicit operator bool();
  A operator&(A);
};

// CHECK: struct A
// CHECK-NEXT: {{^[ ]*operator}} void *();
// CHECK-NEXT: {{^[ ]*explicit}} operator bool();

void bar(void *);

void foo() {
  A a, b;
  bar(a & b);
// CHECK: bar(a & b);
  if (a & b)
// CHECK: if (a & b)
    return;
}
};

namespace {
void test(int i) {
  switch (i) {
    case 1:
      // CHECK: {{\[\[clang::fallthrough\]\]}}
      [[clang::fallthrough]];
    case 2:
      break;
  }
}
}

namespace {
// CHECK: struct {{\[\[gnu::visibility\(\"hidden\"\)\]\]}} S;
struct [[gnu::visibility("hidden")]] S;
}

// CHECK:      struct CXXFunctionalCastExprPrint {
// CHECK-NEXT: } fce = CXXFunctionalCastExprPrint{};
struct CXXFunctionalCastExprPrint {} fce = CXXFunctionalCastExprPrint{};

// CHECK:      struct CXXTemporaryObjectExprPrint {
// CHECK-NEXT:   CXXTemporaryObjectExprPrint();
// CHECK-NEXT: } toe = CXXTemporaryObjectExprPrint{};
struct CXXTemporaryObjectExprPrint { CXXTemporaryObjectExprPrint(); } toe = CXXTemporaryObjectExprPrint{};

namespace PR24872 {
// CHECK: template <typename T> struct Foo : T {
// CHECK: using T::operator-;
template <typename T> struct Foo : T {
  using T::operator-;
};
}

namespace dont_crash_on_auto_vars {
struct T { enum E {X = 12ll }; };
struct S {
  struct  { int I; } ADecl;
  static const auto Y = T::X;
};
//CHECK: static const auto Y = T::X;
constexpr auto var = T::X;
//CHECK: constexpr auto var = T::X;
}
