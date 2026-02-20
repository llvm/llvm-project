// RUN: clang-tidy %s -checks=-*,modernize-redundant-void-arg -- -std=c99 -Wno-strict-prototypes | count 0
// RUN: clang-tidy %s -checks=-*,modernize-redundant-void-arg -- -std=c11 -Wno-strict-prototypes | count 0
// RUN: clang-tidy %s -checks=-*,modernize-redundant-void-arg -- -std=c17 -Wno-strict-prototypes | count 0
// RUN: %check_clang_tidy -std=c23-or-later -check-suffixes=C23 %s modernize-redundant-void-arg %t

#define NULL 0

extern int i;

int foo2() {
  return 0;
}

int j = 1;

int foo(void) {
// CHECK-MESSAGES-C23: :[[@LINE-1]]:9: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-FIXES-C23: int foo() {
  return 0;
}

typedef unsigned int my_uint;

typedef void my_void;

// A function taking void and returning a pointer to function taking void
// and returning int.
int (*returns_fn_void_int(void))(void);
// CHECK-MESSAGES-C23: :[[@LINE-1]]:27: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-MESSAGES-C23: :[[@LINE-2]]:34: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-FIXES-C23: int (*returns_fn_void_int())();

typedef int (*returns_fn_void_int_t(void))(void);
// CHECK-MESSAGES-C23: :[[@LINE-1]]:37: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-MESSAGES-C23: :[[@LINE-2]]:44: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-FIXES-C23: typedef int (*returns_fn_void_int_t())();

int (*returns_fn_void_int(void))(void) {
// CHECK-MESSAGES-C23: :[[@LINE-1]]:27: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-MESSAGES-C23: :[[@LINE-2]]:34: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-FIXES-C23: int (*returns_fn_void_int())() {
  return NULL;
}

// A function taking void and returning a pointer to a function taking void
// and returning a pointer to a function taking void and returning void.
void (*(*returns_fn_returns_fn_void_void(void))(void))(void);
// CHECK-MESSAGES-C23: :[[@LINE-1]]:42: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-MESSAGES-C23: :[[@LINE-2]]:49: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-MESSAGES-C23: :[[@LINE-3]]:56: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-FIXES-C23: void (*(*returns_fn_returns_fn_void_void())())();

typedef void (*(*returns_fn_returns_fn_void_void_t(void))(void))(void);
// CHECK-MESSAGES-C23: :[[@LINE-1]]:52: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-MESSAGES-C23: :[[@LINE-2]]:59: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-MESSAGES-C23: :[[@LINE-3]]:66: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-FIXES-C23: typedef void (*(*returns_fn_returns_fn_void_void_t())())();

void (*(*returns_fn_returns_fn_void_void(void))(void))(void) {
// CHECK-MESSAGES-C23: :[[@LINE-1]]:42: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-MESSAGES-C23: :[[@LINE-2]]:49: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-MESSAGES-C23: :[[@LINE-3]]:56: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-FIXES-C23: void (*(*returns_fn_returns_fn_void_void())())() {
  return NULL;
}

void bar(void) {
// CHECK-MESSAGES-C23: :[[@LINE-1]]:10: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-FIXES-C23: void bar() {
  int i;
  int *pi = NULL;
  void *pv = (void *) pi;
  float f;
  float *fi;
  double d;
  double *pd;
}

void (*f1)(void);
// CHECK-MESSAGES-C23: :[[@LINE-1]]:12: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-FIXES-C23: void (*f1)();
void (*f2)(void) = NULL;
// CHECK-MESSAGES-C23: :[[@LINE-1]]:12: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-FIXES-C23: void (*f2)() = NULL;
void (*f3)(void) = bar;
// CHECK-MESSAGES-C23: :[[@LINE-1]]:12: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-FIXES-C23: void (*f3)() = bar;
void (*fa)();
void (*fb)() = NULL;
void (*fc)() = bar;

typedef void (function_ptr)(void);
// CHECK-MESSAGES-C23: :[[@LINE-1]]:29: warning: redundant void argument list [modernize-redundant-void-arg]
// CHECK-FIXES-C23: typedef void (function_ptr)();
