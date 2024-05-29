// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s --implicit-check-not='call{{.*}}dtor'
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-CXX23,CHECK-CXX23-NEXT,CHECK-CXX23-LABEL

namespace std {
  typedef decltype(sizeof(int)) size_t;

  template <class E>
  struct initializer_list {
    const E *begin;
    size_t   size;
    initializer_list() : begin(nullptr), size(0) {}
  };

  template <typename E>
  struct list {
    list() {}
    ~list() {}
    E *begin();
    E *end();
    const E *begin() const;
    const E *end() const;
  };

  template <typename E>
  struct vector {
    vector() {}
    vector(std::initializer_list<E>) {}
    ~vector() {}
    E *begin();
    E *end();
    const E *begin() const;
    const E *end() const;
  };

  template <typename T>
  struct lock_guard {
    lock_guard(T) {}
    ~lock_guard() {}
  };

  struct mutex {};
} // namespace std

void then();

struct dtor {
  ~dtor();
};

dtor ctor();

auto &&lambda = [a = {ctor()}] {};
// CHECK-LABEL: define
// CHECK: call {{.*}}ctor
// CHECK: call {{.*}}atexit{{.*}}global_array_dtor

// CHECK-LABEL: define{{.*}}global_array_dtor
// CHECK: call {{.*}}dtor

// [lifetime extension occurs if the object was obtained by]
//  -- a temporary materialization conversion
// CHECK-LABEL: ref_binding
void ref_binding() {
  // CHECK: call {{.*}}ctor
  auto &&x = ctor();
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}

//  -- ( expression )
// CHECK-LABEL: parens
void parens() {
  // CHECK: call {{.*}}ctor
  auto &&x = ctor();
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}

//  -- subscripting of an array
// CHECK-LABEL: array_subscript_1
void array_subscript_1() {
  using T = dtor[1];
  // CHECK: call {{.*}}ctor
  auto &&x = T{ctor()}[0];
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: array_subscript_2
void array_subscript_2() {
  using T = dtor[1];
  // CHECK: call {{.*}}ctor
  auto &&x = ((dtor*)T{ctor()})[0];
  // CHECK: call {{.*}}dtor
  // CHECK: call {{.*}}then
  then();
  // CHECK: }
}

struct with_member { dtor d; ~with_member(); };
struct with_ref_member { dtor &&d; ~with_ref_member(); };

//  -- a class member access using the . operator [...]
// CHECK-LABEL: member_access_1
void member_access_1() {
  // CHECK: call {{.*}}ctor
  auto &&x = with_member{ctor()}.d;
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}with_member
  // CHECK: }
}
// CHECK-LABEL: member_access_2
void member_access_2() {
  // CHECK: call {{.*}}ctor
  auto &&x = with_ref_member{ctor()}.d;
  // CHECK: call {{.*}}with_ref_member
  // CHECK: call {{.*}}dtor
  // CHECK: call {{.*}}then
  then();
  // CHECK: }
}
// CHECK-LABEL: member_access_3
void member_access_3() {
  // CHECK: call {{.*}}ctor
  auto &&x = (&(const with_member&)with_member{ctor()})->d;
  // CHECK: call {{.*}}with_member
  // CHECK: call {{.*}}then
  then();
  // CHECK: }
}

//  -- a pointer-to-member operation using the .* operator [...]
// CHECK-LABEL: member_ptr_access_1
void member_ptr_access_1() {
  // CHECK: call {{.*}}ctor
  auto &&x = with_member{ctor()}.*&with_member::d;
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}with_member
  // CHECK: }
}
// CHECK-LABEL: member_ptr_access_2
void member_ptr_access_2() {
  // CHECK: call {{.*}}ctor
  auto &&x = (&(const with_member&)with_member{ctor()})->*&with_member::d;
  // CHECK: call {{.*}}with_member
  // CHECK: call {{.*}}then
  then();
  // CHECK: }
}

//  -- a [named] cast [...]
// CHECK-LABEL: static_cast
void test_static_cast() {
  // CHECK: call {{.*}}ctor
  auto &&x = static_cast<dtor&&>(ctor());
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: const_cast
void test_const_cast() {
  // CHECK: call {{.*}}ctor
  auto &&x = const_cast<dtor&&>(ctor());
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: reinterpret_cast
void test_reinterpret_cast() {
  // CHECK: call {{.*}}ctor
  auto &&x = reinterpret_cast<dtor&&>(static_cast<dtor&&>(ctor()));
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: dynamic_cast
void test_dynamic_cast() {
  // CHECK: call {{.*}}ctor
  auto &&x = dynamic_cast<dtor&&>(ctor());
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}

//  -- [explicit cast notation is defined in terms of the above]
// CHECK-LABEL: c_style_cast
void c_style_cast() {
  // CHECK: call {{.*}}ctor
  auto &&x = (dtor&&)ctor();
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: function_style_cast
void function_style_cast() {
  // CHECK: call {{.*}}ctor
  using R = dtor&&;
  auto &&x = R(ctor());
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}

//  -- a conditional operator
// CHECK-LABEL: conditional
void conditional(bool b) {
  // CHECK: call {{.*}}ctor
  // CHECK: call {{.*}}ctor
  auto &&x = b ? (dtor&&)ctor() : (dtor&&)ctor();
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: call {{.*}}dtor
  // CHECK: }
}

//  -- a comma expression
// CHECK-LABEL: comma
void comma() {
  // CHECK: call {{.*}}ctor
  auto &&x = (true, (dtor&&)ctor());
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}


// This applies recursively: if an object is lifetime-extended and contains a
// reference, the referent is also extended.
// CHECK-LABEL: init_capture_ref
void init_capture_ref() {
  // CHECK: call {{.*}}ctor
  auto x = [&a = (const dtor&)ctor()] {};
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: init_capture_ref_indirect
void init_capture_ref_indirect() {
  // CHECK: call {{.*}}ctor
  auto x = [&a = (const dtor&)ctor()] {};
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: init_capture_init_list
void init_capture_init_list() {
  // CHECK: call {{.*}}ctor
  auto x = [a = {ctor()}] {};
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}

void check_dr1815() { // dr1815: yes
#if __cplusplus >= 201402L

  struct A {
    int &&r = 0;
    ~A() {}
  };

  struct B {
    A &&a = A{};
    ~B() {}
  };
  B a = {};
  
  // CHECK: call {{.*}}block_scope_begin_function
  extern void block_scope_begin_function();
  extern void block_scope_end_function();
  block_scope_begin_function();
  {
    // CHECK: call void @_ZZ12check_dr1815vEN1BD1Ev
    // CHECK: call void @_ZZ12check_dr1815vEN1AD1Ev
    B b = {};
  }
  // CHECK: call {{.*}}block_scope_end_function
  block_scope_end_function();

  // CHECK: call {{.*}}some_other_function
  extern void some_other_function();
  some_other_function();
  // CHECK: call void @_ZZ12check_dr1815vEN1BD1Ev
  // CHECK: call void @_ZZ12check_dr1815vEN1AD1Ev
#endif
}

namespace P2718R0 {
namespace basic {
template <typename E> using T2 = std::list<E>;
template <typename E> const T2<E> &f1_temp(const T2<E> &t)  { return t; }
template <typename E> const T2<E> &f2_temp(T2<E> t)         { return t; }
template <typename E> T2<E> g_temp()                        { return T2<E>{}; }

template <typename E>
void foo_dependent_context1() {
  // CHECK-CXX23: void @_ZN7P2718R05basic22foo_dependent_context1IiEEvv()
  // CHECK-CXX23: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZNSt4listIiED1Ev(
  for (auto e : f1_temp(g_temp<E>())) {}  // OK, lifetime of return value of g() extended
}

template <typename E>
void foo_dependent_context2() {
  // CHECK-CXX23: void @_ZN7P2718R05basic22foo_dependent_context2IiEEvv()
  // CHECK-CXX23-NEXT: entry:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R05basic6g_tempIiEESt4listIT_Ev(
  // CHECK-CXX23-NEXT: call {{.*}} @_ZN7P2718R05basic7f2_tempIiEERKSt4listIT_ES4_(
  // CHECK-CXX23-NEXT: call void @_ZNSt4listIiED1Ev(
  // CHECK-CXX23: call {{.*}} @_ZNKSt4listIiE5beginEv(
  // CHECK-CXX23: call {{.*}} @_ZNKSt4listIiE3endEv(
  for (auto e : f2_temp(g_temp<E>())) {}  // undefined behavior
}

template void foo_dependent_context1<int>();
template void foo_dependent_context2<int>();
} // namespace basic

namespace discard_value_expression {
template <typename T>
void f_dependent_context1() {
  std::vector<T> v = { 42, 17, 13 };
  std::mutex m;
  // CHECK-CXX23: void @_ZN7P2718R024discard_value_expression20f_dependent_context1IiEEvv()
  // CHECK-CXX23-LABEL: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZNSt10lock_guardISt5mutexED1Ev(
  for (T x : std::lock_guard<std::mutex>(m), v)  // lock released in C++ 2023
    std::lock_guard<std::mutex> guard(m);  // OK in C++ 2023, now deadlocks
}

template <typename T>
void f_dependent_context2() {
  std::vector<T> v = { 42, 17, 13 };
  std::mutex m;
  // CHECK-CXX23: void @_ZN7P2718R024discard_value_expression20f_dependent_context2IiEEvv()
  // CHECK-CXX23-LABEL: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZNSt10lock_guardISt5mutexED1Ev(
  for (T x : (void)std::lock_guard<std::mutex>(m), v)  // lock released in C++ 2023
    std::lock_guard<std::mutex> guard(m);  // OK in C++ 2023, now deadlocks
}

template <typename T>
void f_dependent_context3() {
  std::vector<T> v = { 42, 17, 13 };
  std::mutex m;
  // CHECK-CXX23: void @_ZN7P2718R024discard_value_expression20f_dependent_context3IiEEvv()
  // CHECK-CXX23-LABEL: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZNSt10lock_guardISt5mutexED1Ev(
  for (T x : static_cast<void>(std::lock_guard<std::mutex>(m)), v)  // lock released in C++ 2023
    std::lock_guard<std::mutex> guard(m);  // OK in C++ 2023, now deadlocks
}

template void f_dependent_context1<int>();
template void f_dependent_context2<int>();
template void f_dependent_context3<int>();
} // namespace discard_value_expression

namespace member_call {
template <typename T>
struct ListWrapper {
  std::list<T> list;
  ListWrapper() {}
  ~ListWrapper() {}
  const T *begin() const { return list.begin(); }
  const T *end() const { return list.end(); }
  ListWrapper& r() { return *this; }
  ListWrapper g() { return ListWrapper(); }
};

template <typename E>
ListWrapper<E> g_temp() { return ListWrapper<E>{}; }

template <typename T>
void member_call_dependent_context() {
  // CHECK-CXX23: void @_ZN7P2718R011member_call29member_call_dependent_contextIiEEvv()
  // CHECK-CXX23-LABEL: for.cond.cleanup: 
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011member_call11ListWrapperIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011member_call11ListWrapperIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011member_call11ListWrapperIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011member_call11ListWrapperIiED1Ev(
  for (auto e : g_temp<T>().r().g().r().g().r().g()) {}
}

template void member_call_dependent_context<int>();
} // namespace member_call

namespace default_arg {
template <typename T>
struct DefaultArg {
  DefaultArg() {}
  DefaultArg(int) {}
  ~DefaultArg() {}
};

template <typename T>
struct C2 : public std::list<T> {
  C2() {}
  C2(int, const C2 &, const DefaultArg<T> &Default = DefaultArg<T>{}) {}
};

template <typename T>
std::list<T> temp_foo(const std::list<T>&, const DefaultArg<T> &Default = DefaultArg<T>{}) {
  return std::list<T>{};
}

template <typename T>
void default_arg_dependent_context1() {
  // CHECK-CXX23: void @_ZN7P2718R011default_arg30default_arg_dependent_context1IiEEvv()
  // CHECK-CXX23-LABEL: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZNSt4listIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg10DefaultArgIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZNSt4listIiED1Ev(
  for (auto e : temp_foo(std::list<T>{})) {}
}

template <typename T>
void default_arg_dependent_context2() {
  // CHECK-CXX23: void @_ZN7P2718R011default_arg30default_arg_dependent_context2IiEEvv()
  // CHECK-CXX23-LABEL: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZNSt4listIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg10DefaultArgIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZNSt4listIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg10DefaultArgIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZNSt4listIiED1Ev(
  for (auto e : temp_foo(temp_foo(std::list<T>{}))) {}
}

template <typename T>
void default_arg_dependent_context3() {
  // CHECK-CXX23: void @_ZN7P2718R011default_arg30default_arg_dependent_context3IiEEvv()
  // CHECK-CXX23-LABEL: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg2C2IiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg10DefaultArgIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg2C2IiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg10DefaultArgIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg2C2IiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg10DefaultArgIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg2C2IiED1Ev(

  for (auto e : C2<T>(0, C2<T>(0, C2<T>(0, C2<T>())))) {}
}

template void default_arg_dependent_context1<int>();
template void default_arg_dependent_context2<int>();
template void default_arg_dependent_context3<int>();
} // namespace default_arg

namespace basic {
using T = std::list<int>;
const T& f1(const T& t) { return t; }
const T& f2(T t)        { return t; }
T g()                   { return T{}; }

void foo1() {
  // CHECK-CXX23: void @_ZN7P2718R05basic4foo1Ev()
  // CHECK-CXX23: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZNSt4listIiED1Ev(
  for (auto e : f1(g())) {}  // OK, lifetime of return value of g() extended
}

void foo2() {
  // CHECK-CXX23: void @_ZN7P2718R05basic4foo2Ev()
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R05basic1gEv(
  // CHECK-CXX23-NEXT: call {{.*}} @_ZN7P2718R05basic2f2ESt4listIiE(
  // CHECK-CXX23-NEXT: call void @_ZNSt4listIiED1Ev(
  for (auto e : f2(g())) {}  // undefined behavior
}
} // namespace basic

namespace discard_value_expression {
void f1() {
  std::vector<int> v = { 42, 17, 13 };
  std::mutex m;
  // CHECK-CXX23: void @_ZN7P2718R024discard_value_expression2f1Ev()
  // CHECK-CXX23-LABEL: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZNSt10lock_guardISt5mutexED1Ev( 
  for (int x : std::lock_guard<std::mutex>(m), v)  // lock released in C++ 2023
    std::lock_guard<std::mutex> guard(m);  // OK in C++ 2023, now deadlocks
}

void f2() {
  std::vector<int> v = { 42, 17, 13 };
  std::mutex m;
  // CHECK-CXX23: void @_ZN7P2718R024discard_value_expression2f2Ev()
  // CHECK-CXX23-LABEL: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZNSt10lock_guardISt5mutexED1Ev(
  for (int x : (void)std::lock_guard<std::mutex>(m), v)  // lock released in C++ 2023
    std::lock_guard<std::mutex> guard(m);  // OK in C++ 2023, now deadlocks
}

void f3() {
  std::vector<int> v = { 42, 17, 13 };
  std::mutex m;
  // CHECK-CXX23: void @_ZN7P2718R024discard_value_expression2f3Ev()
  // CHECK-CXX23-LABEL: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZNSt10lock_guardISt5mutexED1Ev(
  for (int x : static_cast<void>(std::lock_guard<std::mutex>(m)), v)  // lock released in C++ 2023
    std::lock_guard<std::mutex> guard(m);  // OK in C++ 2023, now deadlocks
}
} // namespace discard_value_expression

namespace member_call {
using A = ListWrapper<int>;

A g() { return A(); }
const A &f1(const A &t) { return t; }

void member_call() {
  // CHECK-CXX23: void @_ZN7P2718R011member_call11member_callEv()
  // CHECK-CXX23-LABEL: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011member_call11ListWrapperIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011member_call11ListWrapperIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011member_call11ListWrapperIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011member_call11ListWrapperIiED1Ev(
  for (auto e : g().r().g().r().g().r().g()) {}
}
} // namespace member_call

namespace default_arg {
using A = std::list<int>;
using DefaultA = DefaultArg<int>;
struct C : public A {
  C() {}
  C(int, const C &, const DefaultA & = DefaultA()) {}
};

A foo(const A&, const DefaultA &Default = DefaultA()) {
  return A();
}

int (&some_func(const A & = A{}))[3];

void default_arg1() {
  // CHECK-CXX23: void @_ZN7P2718R011default_arg12default_arg1Ev()
  // CHECK-CXX23-LABEL: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZNSt4listIiED1Ev(
  for (auto e : some_func()) {}
}

void default_arg2() {
  // CHECK-CXX23: void @_ZN7P2718R011default_arg12default_arg2Ev()
  // CHECK-CXX23-LABEL: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZNSt4listIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg10DefaultArgIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZNSt4listIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg10DefaultArgIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZNSt4listIiED1Ev(
  for (auto e : some_func(foo(foo(A())))) {}
}

void default_arg3() {
  // CHECK-CXX23: void @_ZN7P2718R011default_arg12default_arg3Ev()
  // CHECK-CXX23-LABEL: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg1CD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg10DefaultArgIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg1CD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg10DefaultArgIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg1CD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg10DefaultArgIiED1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R011default_arg1CD1Ev(
  for (auto e : C(0, C(0, C(0, C())))) {}
}
} // namespace default_arg
} // namespace P2718R0

