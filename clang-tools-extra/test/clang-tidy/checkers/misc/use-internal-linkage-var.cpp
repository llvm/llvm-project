// RUN: %check_clang_tidy %s misc-use-internal-linkage %t -- -- -I%S/Inputs/use-internal-linkage
// RUN: %check_clang_tidy %s misc-use-internal-linkage %t -- \
// RUN:   -config="{CheckOptions: {misc-use-internal-linkage.FixMode: 'UseStatic'}}"  -- -I%S/Inputs/use-internal-linkage

#include "var.h"

int global;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: variable 'global' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static int global;

template<class T>
T global_template;
// CHECK-MESSAGES: :[[@LINE-1]]:3: warning: variable 'global_template' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static T global_template;

int const* ptr_const_star;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: variable 'ptr_const_star' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static int const* ptr_const_star;

const int* const_ptr_star;
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: variable 'const_ptr_star' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static const int* const_ptr_star;

const volatile int* const_volatile_ptr_star;
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: variable 'const_volatile_ptr_star' can be made static or moved into an anonymous namespace to enforce internal linkage
// CHECK-FIXES: static const volatile int* const_volatile_ptr_star;

int global_header;

extern int global_extern;

static int global_static;

thread_local int global_thread_local;

namespace {
static int global_anonymous_ns;
namespace NS {
static int global_anonymous_ns;
}
}

static void f(int para) {
  int local;
  static int local_static;
  thread_local int local_thread_local;
}

struct S {
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: struct 'S' can be moved into an anonymous namespace to enforce internal linkage
  int m1;
  static int m2;
};
int S::m2;

extern "C" {
int global_in_extern_c_1;
}

extern "C" int global_in_extern_c_2;

const int const_global = 123;
constexpr int constexpr_global = 123;
