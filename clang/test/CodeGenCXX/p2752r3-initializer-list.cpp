// RUN: %clang_cc1 %std_cxx11- -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm -o - --embed-dir=%S/Inputs -Wno-c23-extensions %s | FileCheck %s
// RUN: %clang_cc1 %std_cxx11- -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -S -o - --embed-dir=%S/Inputs -Wno-c23-extensions %s | FileCheck --check-prefix=ASM %s

// CHECK-DAG: @[[DOUBLE_INIT:[.A-Za-z0-9_$]+]] = private unnamed_addr constant [3 x double] [double 1.000000e+00, double 2.000000e+00, double 3.000000e+00], align 8
// CHECK-DAG: @[[EMBED_INIT:[.A-Za-z0-9_$]+]] = private unnamed_addr constant [2 x i8] c"jk", align 1
// CHECK-DAG: @[[HUNDO_INIT:[.A-Za-z0-9_$]+]] = private unnamed_addr constant [100 x i32]
// CHECK-DAG: @[[INT_INIT:[.A-Za-z0-9_$]+]] = private unnamed_addr constant [3 x i32] [i32 1, i32 2, i32 3], align 4

namespace std {
using size_t = decltype(sizeof(int));

template <class E> class initializer_list {
  const E *begin_;
  size_t size_;

public:
  constexpr initializer_list() : begin_(nullptr), size_(0) {}
  constexpr initializer_list(const E *begin, size_t size)
      : begin_(begin), size_(size) {}
  constexpr const E *begin() const { return begin_; }
  constexpr size_t size() const { return size_; }
};
} // namespace std

namespace example12 {
void f(std::initializer_list<double> il);

void g(float x) {
  // CHECK-LABEL: define{{.*}} void @_ZN9example121gEf(
  // CHECK: alloca [3 x double],
  // CHECK: fpext float
  // CHECK: call void @_ZN9example121fESt16initializer_listIdE(
  f({1, x, 3});
}

void h() {
  // CHECK-LABEL: define{{.*}} void @_ZN9example121hEv(
  // CHECK-NOT: alloca [3 x double]
  // CHECK-NOT: llvm.memcpy
  // CHECK: store ptr @[[DOUBLE_INIT]], ptr %{{.*}}, align 8
  // CHECK: call void @_ZN9example121fESt16initializer_listIdE(
  f({1, 2, 3});
}
} // namespace example12

namespace embed_example {
void bytes(std::initializer_list<unsigned char>);

void f() {
  // CHECK-LABEL: define{{.*}} void @_ZN13embed_example1fEv(
  // CHECK-NOT: alloca [2 x i8]
  // CHECK-NOT: llvm.memcpy
  // CHECK: store ptr @[[EMBED_INIT]], ptr %{{.*}}, align 8
  // CHECK: call void @_ZN13embed_example5bytesESt16initializer_listIhE(
  bytes({
#embed <jk.txt>
  });
}
} // namespace embed_example

namespace large_constant_list {
#define TEN 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
#define HUNDO TEN, TEN, TEN, TEN, TEN, TEN, TEN, TEN, TEN, TEN

void f(std::initializer_list<int>);

void g() {
  // CHECK-LABEL: define{{.*}} void @_ZN19large_constant_list1gEv(
  // CHECK-NOT: alloca [100 x i32]
  // CHECK-NOT: llvm.memcpy
  // CHECK: store ptr @[[HUNDO_INIT]], ptr %{{.*}}, align 8
  // CHECK: call void @_ZN19large_constant_list1fESt16initializer_listIiE(
  f({HUNDO});
}
} // namespace large_constant_list

namespace shared_static_lists {
void f(std::initializer_list<int>);

void g() {
  // CHECK-LABEL: define{{.*}} void @_ZN19shared_static_lists1gEv(
  // CHECK-NOT: alloca [3 x i32]
  // CHECK-NOT: llvm.memcpy
  // CHECK: store ptr @[[INT_INIT]], ptr %{{.*}}, align 8
  // CHECK: call void @_ZN19shared_static_lists1fESt16initializer_listIiE(
  // CHECK: store ptr @[[INT_INIT]], ptr %{{.*}}, align 8
  // CHECK: call void @_ZN19shared_static_lists1fESt16initializer_listIiE(
  f({1, 2, 3});
  f({1, 2, 3});
}
} // namespace shared_static_lists

namespace mergeable_static_list {
void f(std::initializer_list<int>);

void g() {
  // ASM: .section .rodata.cst16,"aM",@progbits,16
  // ASM: .long 1
  // ASM-NEXT: .long 2
  // ASM-NEXT: .long 3
  // ASM-NEXT: .long 4
  f({1, 2, 3, 4});
}
} // namespace mergeable_static_list

namespace destructor_side_effects {
extern "C" int printf(const char *, ...);

struct C6 {
  constexpr C6(int) {}
  ~C6() { printf(" X"); }
};

void f6(std::initializer_list<C6>) {}

void test() {
  // CHECK-LABEL: define{{.*}} void @_ZN23destructor_side_effects4testEv(
  // CHECK: call void @_ZN23destructor_side_effects2f6ESt16initializer_listINS_2C6EE(
  // CHECK: call void @_ZN23destructor_side_effects2C6D1Ev(
  // CHECK: call void @_ZN23destructor_side_effects2f6ESt16initializer_listINS_2C6EE(
  // CHECK: call void @_ZN23destructor_side_effects2C6D1Ev(
  f6({1, 2, 3});
  f6({1, 2, 3});
}
} // namespace destructor_side_effects

namespace mutable_members {
struct S {
  constexpr S(int i) : i(i) {}
  mutable int i;
};

void f(std::initializer_list<S> il) {
  if (il.begin()->i != 1)
    throw;
  il.begin()->i = 4;
}

void test() {
  // CHECK-LABEL: define{{.*}} void @_ZN15mutable_members4testEv(
  // CHECK: alloca [3 x %"struct.mutable_members::S"],
  // CHECK: call void @_ZN15mutable_members1fESt16initializer_listINS_1SEE(
  for (int i = 0; i < 2; ++i)
    f({1, 2, 3});
}
} // namespace mutable_members
