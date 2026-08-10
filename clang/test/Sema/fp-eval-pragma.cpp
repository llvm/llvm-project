// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu -emit-llvm -o - -verify %s
//
// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu -emit-llvm -o - -verify %s \
// RUN: -ffp-eval-method=source
//
// RUN: %clang_cc1 -fexperimental-strict-floating-point \
// RUN: -triple x86_64-linux-gnu -emit-llvm -o - -verify %s \
// RUN: -ffp-eval-method=double

extern "C" int printf(const char *, ...);

void foo1() {
  printf("FP: %d\n", __FLT_EVAL_METHOD__);
}

void apply_pragma() {
  // expected-note@+1{{#pragma entered here}}
#pragma clang fp eval_method(double)
  // expected-error@+1{{'__FLT_EVAL_METHOD__' cannot be expanded inside a scope containing '#pragma clang fp eval_method'}}
  printf("FP: %d\n", __FLT_EVAL_METHOD__);
}

int foo2() {
  apply_pragma();
  return 0;
}

void apply_pragma_with_wrong_value() {
  // expected-error@+1{{unexpected argument 'value' to '#pragma clang fp eval_method'; expected 'source', 'double' or 'extended'}}
#pragma clang fp eval_method(value)
}

int foo3() {
  apply_pragma_with_wrong_value();
  return 0;
}

void foo() {
  auto a = __FLT_EVAL_METHOD__;
  {
    // expected-note@+1{{#pragma entered here}}
#pragma clang fp eval_method(double)
    // expected-error@+1{{'__FLT_EVAL_METHOD__' cannot be expanded inside a scope containing '#pragma clang fp eval_method'}}
    auto b = __FLT_EVAL_METHOD__;
  }
  auto c = __FLT_EVAL_METHOD__;
}

void func() {
  {
    {
#pragma clang fp eval_method(source)
    }
    int i = __FLT_EVAL_METHOD__; // ok, not in a scope changed by the pragma
  }
  {
    // expected-note@+1{{#pragma entered here}}
#pragma clang fp eval_method(source)
    // expected-error@+1{{'__FLT_EVAL_METHOD__' cannot be expanded inside a scope containing '#pragma clang fp eval_method'}}
    int i = __FLT_EVAL_METHOD__;
  }
}

float G;

int f(float x, float y, float z) {
  G = x * y + z;
  return __FLT_EVAL_METHOD__;
}

int foo(int flag, float x, float y, float z) {
  if (flag) {
    // expected-note@+1{{#pragma entered here}}
#pragma clang fp eval_method(double)
    G = x + y + z;
    // expected-error@+1{{'__FLT_EVAL_METHOD__' cannot be expanded inside a scope containing '#pragma clang fp eval_method'}}
    return __FLT_EVAL_METHOD__;
  } else {
    // expected-note@+1{{#pragma entered here}}
#pragma clang fp eval_method(extended)
    G = x + y + z;
    // expected-error@+1{{'__FLT_EVAL_METHOD__' cannot be expanded inside a scope containing '#pragma clang fp eval_method'}}
    return __FLT_EVAL_METHOD__;
  }
}

#if __FLT_EVAL_METHOD__ == 1
#endif
#pragma clang fp eval_method(source)

// expected-note@+1{{#pragma entered here}}
#pragma clang fp eval_method(double)
// expected-error@+1{{'__FLT_EVAL_METHOD__' cannot be expanded inside a scope containing '#pragma clang fp eval_method'}}
#if __FLT_EVAL_METHOD__ == 1
#endif
