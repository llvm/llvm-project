// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++14 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++14 -emit-llvm -o - %s -fexperimental-new-constant-interpreter | FileCheck %s

int b = 10;
const int f = (__builtin_assume_dereferenceable((char*)&b + 1, 3), 12);
int use_f = f;

constexpr int g = 20;
const int h = (__builtin_assume_dereferenceable((char*)&g + 1, 2), 42);
int use_h = h;

constexpr char arr[] = "hello";
constexpr const char* ptr = arr + 1;
constexpr int fully_constexpr() {
  __builtin_assume_dereferenceable(ptr, 2);
  return 100;
}
constexpr int i = fully_constexpr();
int use_i = i;

void test_nullptr() {
  __builtin_assume_dereferenceable(nullptr, 0);
}

void test_zero_size() {
  int x = 10;
  __builtin_assume_dereferenceable(&x, 0);
}

void test_function_ptr() {
  __builtin_assume_dereferenceable((void*)&test_zero_size, 8);
}

// CHECK: @use_i = global i32 100
//
// CHECK: @{{_Z[0-9]+}}test_nullptrv
// CHECK: call void @llvm.assume(i1 true) [ "dereferenceable"(ptr null, i64 0) ]
//
// CHECK: @{{_Z[0-9]+}}test_zero_sizev
// CHECK: call void @llvm.assume(i1 true) [ "dereferenceable"(ptr {{%.*}}, i64 0) ]
//
// CHECK: @{{_Z[0-9]+}}test_function_ptrv
// CHECK: call void @llvm.assume(i1 true) [ "dereferenceable"(ptr @{{_Z[0-9]+}}test_zero_sizev, i64 8) ]
//
// CHECK: __cxx_global_var_init
// CHECK: call void @llvm.assume(i1 true) [ "dereferenceable"(ptr getelementptr inbounds (i8, ptr @b, i64 1), i64 3) ]
//
// CHECK: __cxx_global_var_init
// CHECK: call void @llvm.assume(i1 true) [ "dereferenceable"(ptr getelementptr inbounds (i8, ptr @{{_ZL[0-9]+}}g, i64 1), i64 2) ]
// CHECK: store i32 42, ptr @{{_ZL[0-9]+}}h
