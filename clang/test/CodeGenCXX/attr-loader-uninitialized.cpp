// UNSUPPORTED: system-windows
// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// CHECK: @defn = global i32 undef
int defn  [[clang::loader_uninitialized]];

// CHECK: @_ZL11defn_static = internal global i32 undef
static int defn_static [[clang::loader_uninitialized]] __attribute__((used));

// CHECK: @_ZZ4funcvE4data = internal global i32 undef
int* func(void)
{
  static int data [[clang::loader_uninitialized]];
  return &data;
}

class trivial
{
  float x;
};

// CHECK: @ut = global %class.trivial undef
trivial ut [[clang::loader_uninitialized]];

// CHECK: @arr = global [32 x double] undef
double arr[32] __attribute__((loader_uninitialized));

// Defining as arr2[] [[clang..]] raises the error: attribute cannot be applied to types
// CHECK: @arr2 = global [4 x double] undef
double arr2 [[clang::loader_uninitialized]] [4];
