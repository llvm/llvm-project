// RUN: %clang_analyze_cc1 -std=c++11 -fblocks -analyzer-checker=core,debug.ExprInspection -verify %s -o %t.report
// RUN: %clang_analyze_cc1 -std=c++11 -fblocks -analyzer-checker=core,debug.ExprInspection -DEMULATE_LIBSTDCPP -verify %s -o %t.report

// We do NOT model libcxx03 implementation, but the analyzer should still
// not crash.
// RUN: %clang_analyze_cc1 -std=c++11 -fblocks -analyzer-checker=core,debug.ExprInspection -DEMULATE_LIBCXX03 -verify %s -o %t.report
// RUN: %clang_analyze_cc1 -std=c++11 -fblocks -analyzer-checker=core,debug.ExprInspection -DEMULATE_LIBCXX03 -DEMULATE_LIBSTDCPP -verify %s -o %t.report
// RUN: rm -rf %t.report

void clang_analyzer_eval(bool);

// Faking std::call_once implementation.
namespace std {

// Fake std::function implementation.
template <typename>
class function;
class function_base {
 public:
  long field;
};
template <typename R, typename... P>
class function<R(P...)> : function_base {
 public:
   R operator()(P...) const {

     // Read from a super-class necessary to reproduce a crash.
     bool a = field;
   }
};

#ifndef EMULATE_LIBSTDCPP
typedef struct once_flag_s {
  unsigned long __state_ = 0;
} once_flag;
#else
typedef struct once_flag_s {
  int _M_once = 0;
} once_flag;
#endif

#ifndef EMULATE_LIBCXX03
template <class Callable, class... Args>
void call_once(once_flag &o, Callable&& func, Args&&... args) {};
#else
template <class Callable, class... Args> // libcxx03 call_once
void call_once(once_flag &o, Callable func, Args&&... args) {};
#endif

} // namespace std

// Check with Lambdas.
void test_called_warning() {
  std::once_flag g_initialize;
  int z;

  std::call_once(g_initialize, [&] {
    int *x = nullptr;
#ifndef EMULATE_LIBCXX03
    int y = *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
#endif
    z = 200;
  });
}

void test_called_on_path_inside_no_warning() {
  std::once_flag g_initialize;

  int *x = nullptr;
  int y = 100;
  int z;

  std::call_once(g_initialize, [&] {
    z = 200;
    x = &z;
  });

#ifndef EMULATE_LIBCXX03
  *x = 100; // no-warning
  clang_analyzer_eval(z == 100); // expected-warning{{TRUE}}
#endif
}

void test_called_on_path_no_warning() {
  std::once_flag g_initialize;

  int *x = nullptr;
  int y = 100;

  std::call_once(g_initialize, [&] {
    x = &y;
  });

#ifndef EMULATE_LIBCXX03
  *x = 100; // no-warning
#else
  *x = 100; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
#endif
}

void test_called_on_path_warning() {
  std::once_flag g_initialize;

  int y = 100;
  int *x = &y;

  std::call_once(g_initialize, [&] {
    x = nullptr;
  });

#ifndef EMULATE_LIBCXX03
  *x = 100; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
#endif
}

void test_called_once_warning() {
  std::once_flag g_initialize;

  int *x = nullptr;
  int y = 100;

  std::call_once(g_initialize, [&] {
    x = nullptr;
  });

  std::call_once(g_initialize, [&] {
    x = &y;
  });

#ifndef EMULATE_LIBCXX03
  *x = 100; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
#endif
}

void test_called_once_no_warning() {
  std::once_flag g_initialize;

  int *x = nullptr;
  int y = 100;

  std::call_once(g_initialize, [&] {
    x = &y;
  });

  std::call_once(g_initialize, [&] {
    x = nullptr;
  });

#ifndef EMULATE_LIBCXX03
  *x = 100; // no-warning
#endif
}

static int global = 0;
void funcPointer() {
  global = 1;
}

void test_func_pointers() {
  static std::once_flag flag;
  std::call_once(flag, &funcPointer);
#ifndef EMULATE_LIBCXX03
  clang_analyzer_eval(global == 1); // expected-warning{{TRUE}}
#endif
}

template <class _Fp>
class function; // undefined
template <class _Rp, class... _ArgTypes>
struct function<_Rp(_ArgTypes...)> {
  _Rp operator()(_ArgTypes...) const {};
  template <class _Fp>
  function(_Fp) {};
};

// Note: currently we do not support calls to std::function,
// but the analyzer should not crash either.
void test_function_objects_warning() {
  int x = 0;
  int *y = &x;

  std::once_flag flag;

  function<void()> func = [&]() {
    y = nullptr;
  };

  std::call_once(flag, func);

  func();
  int z = *y;
}

void test_param_passing_lambda() {
  std::once_flag flag;
  int x = 120;
  int y = 0;

  std::call_once(flag, [&](int p) {
    y = p;
  },
                 x);

#ifndef EMULATE_LIBCXX03
  clang_analyzer_eval(y == 120); // expected-warning{{TRUE}}
#endif
}

void test_param_passing_lambda_false() {
  std::once_flag flag;
  int x = 120;

  std::call_once(flag, [&](int p) {
    x = 0;
  },
                 x);

#ifndef EMULATE_LIBCXX03
  clang_analyzer_eval(x == 120); // expected-warning{{FALSE}}
#endif
}

void test_param_passing_stored_lambda() {
  std::once_flag flag;
  int x = 120;
  int y = 0;

  auto lambda = [&](int p) {
    y = p;
  };

  std::call_once(flag, lambda, x);
#ifndef EMULATE_LIBCXX03
  clang_analyzer_eval(y == 120); // expected-warning{{TRUE}}
#endif
}

void test_multiparam_passing_lambda() {
  std::once_flag flag;
  int x = 120;

  std::call_once(flag, [&](int a, int b, int c) {
    x = a + b + c;
  },
                 1, 2, 3);

#ifndef EMULATE_LIBCXX03
  clang_analyzer_eval(x == 120); // expected-warning{{FALSE}}
  clang_analyzer_eval(x == 6); // expected-warning{{TRUE}}
#endif
}

static int global2 = 0;
void test_param_passing_lambda_global() {
  std::once_flag flag;
  global2 = 0;
  std::call_once(flag, [&](int a, int b, int c) {
    global2 = a + b + c;
  },
                 1, 2, 3);
#ifndef EMULATE_LIBCXX03
  clang_analyzer_eval(global2 == 6); // expected-warning{{TRUE}}
#endif
}

static int global3 = 0;
void funcptr(int a, int b, int c) {
  global3 = a + b + c;
}

void test_param_passing_funcptr() {
  std::once_flag flag;
  global3 = 0;

  std::call_once(flag, &funcptr, 1, 2, 3);

#ifndef EMULATE_LIBCXX03
  clang_analyzer_eval(global3 == 6); // expected-warning{{TRUE}}
#endif
}

void test_blocks() {
  global3 = 0;
  std::once_flag flag;
  std::call_once(flag, ^{
    global3 = 120;
  });
#ifndef EMULATE_LIBCXX03
  clang_analyzer_eval(global3 == 120); // expected-warning{{TRUE}}
#endif
}

int call_once() {
  return 5;
}

void test_non_std_call_once() {
  int x = call_once();
#ifndef EMULATE_LIBCXX03
  clang_analyzer_eval(x == 5); // expected-warning{{TRUE}}
#endif
}

namespace std {
template <typename d, typename e>
void call_once(d, e);
}
void g();
void test_no_segfault_on_different_impl() {
#ifndef EMULATE_LIBCXX03
  std::call_once(g, false); // no-warning
#endif
}

void test_lambda_refcapture() {
  static std::once_flag flag;
  int a = 6;
  std::call_once(flag, [&](int &a) { a = 42; }, a);
#ifndef EMULATE_LIBCXX03
  clang_analyzer_eval(a == 42); // expected-warning{{TRUE}}
#endif
}

void test_lambda_refcapture2() {
  static std::once_flag flag;
  int a = 6;
  std::call_once(flag, [=](int &a) { a = 42; }, a);
#ifndef EMULATE_LIBCXX03
  clang_analyzer_eval(a == 42); // expected-warning{{TRUE}}
#endif
}

void test_lambda_fail_refcapture() {
  static std::once_flag flag;
  int a = 6;
  std::call_once(flag, [=](int a) { a = 42; }, a);
#ifndef EMULATE_LIBCXX03
  clang_analyzer_eval(a == 42); // expected-warning{{FALSE}}
#endif
}

void mutator(int &param) {
  param = 42;
}
void test_reftypes_funcptr() {
  static std::once_flag flag;
  int a = 6;
  std::call_once(flag, &mutator, a);
#ifndef EMULATE_LIBCXX03
  clang_analyzer_eval(a == 42); // expected-warning{{TRUE}}
#endif
}

void fail_mutator(int param) {
  param = 42;
}
void test_mutator_noref() {
  static std::once_flag flag;
  int a = 6;
  std::call_once(flag, &fail_mutator, a);
#ifndef EMULATE_LIBCXX03
  clang_analyzer_eval(a == 42); // expected-warning{{FALSE}}
#endif
}

// Function is implicitly treated as a function pointer
// even when an ampersand is not explicitly set.
void callbackn(int &param) {
  param = 42;
}
void test_implicit_funcptr() {
  int x = 0;
  static std::once_flag flagn;

  std::call_once(flagn, callbackn, x);
#ifndef EMULATE_LIBCXX03
  clang_analyzer_eval(x == 42); // expected-warning{{TRUE}}
#endif
}

int param_passed(int *x) {
  return *x; // no-warning, as std::function is not working yet.
}

void callback_taking_func_ok(std::function<void(int*)> &innerCallback) {
  innerCallback(nullptr);
}

// The provided callback expects an std::function, but instead a pointer
// to a C++ function is provided.
void callback_with_implicit_cast_ok() {
  std::once_flag flag;
  call_once(flag, callback_taking_func_ok, &param_passed);
}

void callback_taking_func(std::function<void()> &innerCallback) {
  innerCallback();
}

// The provided callback expects an std::function, but instead a C function
// name is provided, and C++ implicitly auto-constructs a pointer from it.
void callback_with_implicit_cast() {
  std::once_flag flag;
  call_once(flag, callback_taking_func, callback_with_implicit_cast);
}

std::once_flag another_once_flag;
typedef void (*my_callback_t)(int *);
my_callback_t callback;
int global_int;

void rdar40270582() {
  call_once(another_once_flag, callback, &global_int);
}
