// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++14 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++14 -emit-llvm -o - %s -fexperimental-new-constant-interpreter | FileCheck %s

// Global variables with __builtin_assume_dereferenceable in initializers.
// These generate __cxx_global_var_init functions (checked at end of file).
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

const int j = (__builtin_assume_dereferenceable((int*)0x1234, 4), 200);
int use_j = j;

// CHECK-LABEL: @{{_Z[0-9]+}}test_nullptrv
// CHECK:         call void @llvm.assume(i1 true) [ "dereferenceable"(ptr null, i64 0) ]
void test_nullptr() {
  __builtin_assume_dereferenceable(nullptr, 0);
}

// CHECK-LABEL: @{{_Z[0-9]+}}test_zero_sizev
// CHECK:         call void @llvm.assume(i1 true) [ "dereferenceable"(ptr %{{.*}}, i64 0) ]
void test_zero_size() {
  int x = 10;
  __builtin_assume_dereferenceable(&x, 0);
}

// CHECK-LABEL: @{{_Z[0-9]+}}test_function_ptrv
// CHECK:         call void @llvm.assume(i1 true) [ "dereferenceable"(ptr @{{_Z[0-9]+}}test_zero_sizev, i64 8) ]
void test_function_ptr() {
  __builtin_assume_dereferenceable((void*)&test_zero_size, 8);
}

// CHECK-LABEL: @{{_Z[0-9]+}}test_integral_ptrv
// CHECK:         call void @llvm.assume(i1 true) [ "dereferenceable"(ptr inttoptr (i64 4660 to ptr), i64 4) ]
void test_integral_ptr() {
  __builtin_assume_dereferenceable((int*)0x1234, 4);
}

// Global variable initialization checks for f, h, j above.
// CHECK: call void @llvm.assume(i1 true) [ "dereferenceable"(ptr getelementptr inbounds (i8, ptr @b, i64 1), i64 3) ]
// CHECK: store i32 12, ptr @{{_ZL[0-9]+}}f, align 4

// CHECK: call void @llvm.assume(i1 true) [ "dereferenceable"(ptr getelementptr inbounds (i8, ptr @{{_ZL[0-9]+}}g, i64 1), i64 2) ]
// CHECK: store i32 42, ptr @{{_ZL[0-9]+}}h, align 4

// CHECK: call void @llvm.assume(i1 true) [ "dereferenceable"(ptr inttoptr (i64 4660 to ptr), i64 4) ]
// CHECK: store i32 200, ptr @{{_ZL[0-9]+}}j, align 4
