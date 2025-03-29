// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -fsyntax-only -DNOERROR %s
// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -fsyntax-only -DNOERROR %s -fexperimental-new-constant-interpreter

// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -fsyntax-only \
// RUN: -x c++ -DCPP -DNOERROR %s
// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -fsyntax-only \
// RUN: -x c++ -DCPP -DNOERROR %s -fexperimental-new-constant-interpreter

// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -fsyntax-only \
// RUN: -ffp-eval-method=double -DNOERROR %s
// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -fsyntax-only \
// RUN: -ffp-eval-method=double -DNOERROR %s -fexperimental-new-constant-interpreter

// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -fsyntax-only -x c++ -DCPP \
// RUN: -ffp-eval-method=double -DNOERROR %s
// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -fsyntax-only -x c++ -DCPP \
// RUN: -ffp-eval-method=double -DNOERROR %s -fexperimental-new-constant-interpreter


// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -fsyntax-only \
// RUN: -ffp-eval-method=source %s
// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -fsyntax-only \
// RUN: -ffp-eval-method=source %s -fexperimental-new-constant-interpreter

// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -fsyntax-only -x c++ -DCPP \
// RUN: -ffp-eval-method=source %s
// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -fsyntax-only -x c++ -DCPP \
// RUN: -ffp-eval-method=source %s -fexperimental-new-constant-interpreter

// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -fsyntax-only \
// RUN: -ffp-eval-method=extended %s
// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -fsyntax-only \
// RUN: -ffp-eval-method=extended %s -fexperimental-new-constant-interpreter

// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -fsyntax-only -x c++ -DCPP \
// RUN: -ffp-eval-method=extended %s
// RUN: %clang_cc1 -verify -triple x86_64-linux-gnu -fsyntax-only -x c++ -DCPP \
// RUN: -ffp-eval-method=extended %s -fexperimental-new-constant-interpreter

#ifdef NOERROR
// expected-no-diagnostics
typedef float float_t;
typedef double double_t;
#else
#ifdef CPP
typedef float float_t; //expected-error 9 {{cannot use type 'float_t' within '#pragma clang fp eval_method'; type is set according to the default eval method for the translation unit}}
  
typedef double double_t; //expected-error 9 {{cannot use type 'double_t' within '#pragma clang fp eval_method'; type is set according to the default eval method for the translation unit}}
#else
typedef float float_t; //expected-error 7 {{cannot use type 'float_t' within '#pragma clang fp eval_method'; type is set according to the default eval method for the translation unit}}
  
typedef double double_t; //expected-error 7 {{cannot use type 'double_t' within '#pragma clang fp eval_method'; type is set according to the default eval method for the translation unit}}
#endif
#endif

float foo1() {
#pragma clang fp eval_method(double)
  float a;
  double b;
  return a - b;
}
  
float foo2() {
#pragma clang fp eval_method(double)
  float_t a; 
  double_t b; 
  return a - b;
}
  
void foo3() {
#pragma clang fp eval_method(double)
  char buff[sizeof(float_t)];
  char bufd[sizeof(double_t)];
  buff[1] = bufd[2];
}
  
float foo4() {
#pragma clang fp eval_method(double)
  typedef float_t FT;
  typedef double_t DT;
  FT a;
  DT b;
  return a - b;
}
  
int foo5() {
#pragma clang fp eval_method(double)
  int t = _Generic( 1.0L, float_t:1, default:0);
  int v = _Generic( 1.0L, double_t:1, default:0);
  return t;
}

void foo6() {
#pragma clang fp eval_method(double)
  float f = (float_t)1; 
  double d = (double_t)2; 
}
  
void foo7() {
#pragma clang fp eval_method(double)
  float c1 = (float_t)12;
  double c2 = (double_t)13;
}
  
float foo8() {
#pragma clang fp eval_method(double)
  extern float_t f;
  extern double_t g;
  return f-g;
}

#ifdef CPP
void foo9() {
#pragma clang fp eval_method(double)
  auto resf = [](float_t f) { return f; };
  auto resd = [](double_t g) { return g; };
}

void foo10() {
#pragma clang fp eval_method(double)
  using Ft = float_t;
  using Dt = double_t;
  Ft a;
  Dt b;
}
#endif
 
