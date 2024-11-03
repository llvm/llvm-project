// RUN: %clang_cc1 -x c -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// RUN: %clang_cc1 -x c -std=c89 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x c -std=gnu89 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x c -std=iso9899:1990 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// RUN: %clang_cc1 -x c -std=c17 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x c -std=gnu17 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x c -std=iso9899:2017 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x c -std=c2x -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// RUN: %clang_cc1 -x c++ -std=c++98 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x c++ -std=gnu++98 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x c++ -std=c++17 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x c++ -std=gnu++17 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// RUN: %clang_cc1 -x objective-c++ -std=c++98 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x objective-c++ -std=gnu++98 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x objective-c++ -std=c++17 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x objective-c++ -std=gnu++17 -Wunsafe-buffer-usage -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

// CHECK-NOT: fix-it:

typedef int * Int_ptr_t;
typedef int Int_t;

void local_array_subscript_simple() {
  int tmp;
  int *p;
  const int *q;
  tmp = p[5];
  tmp = q[5];

  Int_ptr_t x;
  Int_ptr_t y;
  Int_t * z;
  Int_t * w;

  tmp = x[5];
  tmp = y[5];
  tmp = z[5];
  tmp = w[5];
}

void local_ptr_to_array() {
  int tmp;
  int n = 10;
  int a[10];
  int b[n];
  int *p = a;
  int *q = b;
  tmp = p[5];
  tmp = q[5];
}

void local_ptr_addrof_init() {
  int var;
  int * q = &var;
  var = q[5];
}

void decl_without_init() {
  int tmp;
  int * p;
  Int_ptr_t q;
  tmp = p[5];
  tmp = q[5];
}

void explict_cast() {
  int tmp;
  int * p;
  tmp = p[5];

  int a;
  char * q = (char *)&a;
  tmp = (int) q[5];

  void * r = &a;
  char * s = (char *) r;
  tmp = (int) s[5];
}
