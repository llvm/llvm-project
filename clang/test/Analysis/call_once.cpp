// RUN: %clang_analyze_cc1 -std=c++11 -fblocks -analyzer-checker=core,debug.ExprInspection -verify %s
// RUN: %clang_analyze_cc1 -std=c++11 -fblocks -analyzer-checker=core,debug.ExprInspection -DEMULATE_LIBSTDCPP -verify %s

void clang_analyzer_eval(bool);

// Faking std::std::call_once implementation.
namespace std {

#ifndef EMULATE_LIBSTDCPP
typedef struct once_flag_s {
  unsigned long __state_ = 0;
} once_flag;
#else
typedef struct once_flag_s {
  int _M_once = 0;
} once_flag;
#endif

template <class Callable, class... Args>
void call_once(once_flag &o, Callable&& func, Args&&... args) {};

} // namespace std

// Check with Lambdas.
void test_called_warning() {
  std::once_flag g_initialize;
  int z;

  std::call_once(g_initialize, [&] {
    int *x = nullptr;
    int y = *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
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

  *x = 100; // no-warning
  clang_analyzer_eval(z == 100); // expected-warning{{TRUE}}
}

void test_called_on_path_no_warning() {
  std::once_flag g_initialize;

  int *x = nullptr;
  int y = 100;

  std::call_once(g_initialize, [&] {
    x = &y;
  });

  *x = 100; // no-warning
}

void test_called_on_path_warning() {
  std::once_flag g_initialize;

  int y = 100;
  int *x = &y;

  std::call_once(g_initialize, [&] {
    x = nullptr;
  });

  *x = 100; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
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

  *x = 100; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
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

  *x = 100; // no-warning
}

static int global = 0;
void funcPointer() {
  global = 1;
}

void test_func_pointers() {
  static std::once_flag flag;
  std::call_once(flag, &funcPointer);
  clang_analyzer_eval(global == 1); // expected-warning{{TRUE}}
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

  clang_analyzer_eval(y == 120); // expected-warning{{TRUE}}
}

void test_param_passing_lambda_false() {
  std::once_flag flag;
  int x = 120;

  std::call_once(flag, [&](int p) {
    x = 0;
  },
                 x);

  clang_analyzer_eval(x == 120); // expected-warning{{FALSE}}
}

void test_param_passing_stored_lambda() {
  std::once_flag flag;
  int x = 120;
  int y = 0;

  auto lambda = [&](int p) {
    y = p;
  };

  std::call_once(flag, lambda, x);
  clang_analyzer_eval(y == 120); // expected-warning{{TRUE}}
}

void test_multiparam_passing_lambda() {
  std::once_flag flag;
  int x = 120;

  std::call_once(flag, [&](int a, int b, int c) {
    x = a + b + c;
  },
                 1, 2, 3);

  clang_analyzer_eval(x == 120); // expected-warning{{FALSE}}
  clang_analyzer_eval(x == 6); // expected-warning{{TRUE}}
}

static int global2 = 0;
void test_param_passing_lambda_global() {
  std::once_flag flag;
  global2 = 0;
  std::call_once(flag, [&](int a, int b, int c) {
    global2 = a + b + c;
  },
                 1, 2, 3);
  clang_analyzer_eval(global2 == 6); // expected-warning{{TRUE}}
}

static int global3 = 0;
void funcptr(int a, int b, int c) {
  global3 = a + b + c;
}

void test_param_passing_funcptr() {
  std::once_flag flag;
  global3 = 0;

  std::call_once(flag, &funcptr, 1, 2, 3);

  clang_analyzer_eval(global3 == 6); // expected-warning{{TRUE}}
}

void test_blocks() {
  global3 = 0;
  std::once_flag flag;
  std::call_once(flag, ^{
    global3 = 120;
  });
  clang_analyzer_eval(global3 == 120); // expected-warning{{TRUE}}
}

int call_once() {
  return 5;
}

void test_non_std_call_once() {
  int x = call_once();
  clang_analyzer_eval(x == 5); // expected-warning{{TRUE}}
}

namespace std {
template <typename d, typename e>
void call_once(d, e);
}
void g();
void test_no_segfault_on_different_impl() {
  std::call_once(g, false); // no-warning
}
