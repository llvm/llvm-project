// RUN: %check_clang_tidy %s misc-use-internal-linkage %t -- -- -I%S/Inputs/use-internal-linkage
// RUN: %check_clang_tidy %s misc-use-internal-linkage %t -- \
// RUN:   -config="{CheckOptions: {misc-use-internal-linkage.FixMode: 'UseStatic'}}"  -- -I%S/Inputs/use-internal-linkage

#include "func.h"

void func(void) {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: function 'func' can be made static to enforce internal linkage
// CHECK-FIXES: static void func(void) {}

void func_header(void) {}
extern void func_extern(void) {}
static void func_static(void) {}

int main(void) {}


#include "var.h"

int global;
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: variable 'global' can be made static to enforce internal linkage
// CHECK-FIXES: static int global;

const int const_global = 123;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: variable 'const_global' can be made static to enforce internal linkage
// CHECK-FIXES: static const int const_global = 123;

int global_header;
extern int global_extern;
static int global_static;
#if __STDC_VERSION__ >= 201112L
_Thread_local int global_thread_local;
#endif
