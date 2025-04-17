// RUN: %clang_cc1 -std=c++14 -emit-llvm -disable-llvm-passes -o - %s -triple x86_64-linux-gnu | FileCheck %s

template<int N> int Fib = Fib<N-2> + Fib<N-1>;
template<> int Fib<0> = 0;
template<> int Fib<1> = 1;
int f = Fib<5>;

template<int N> int Fib2 = Fib2<N-1> + Fib2<N-2>;
template<> int Fib2<0> = 0;
template<> int Fib2<1> = 1;
int f2 = Fib2<5>;

// CHECK: @llvm.global_ctors = appending global [9 x { i32, ptr, ptr }] [
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.4, ptr @_Z3FibILi2EE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.3, ptr @_Z3FibILi3EE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.5, ptr @_Z3FibILi4EE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.2,  ptr @_Z3FibILi5EE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.8, ptr @_Z4Fib2ILi2EE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.9, ptr @_Z4Fib2ILi3EE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.7,  ptr @_Z4Fib2ILi4EE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @__cxx_global_var_init.6, ptr @_Z4Fib2ILi5EE },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_static_init_variable_template.cpp, ptr null }
