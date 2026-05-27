// RUN: %clang_cc1 %std_cxx11- -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm -o - --embed-dir=%S/Inputs -Wno-c23-extensions %s | FileCheck %s

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
  // CHECK: alloca [3 x double],
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 8 {{.*}}, ptr align 8 @constinit, i64 24,
  // CHECK: call void @_ZN9example121fESt16initializer_listIdE(
  f({1, 2, 3});
}
} // namespace example12

namespace embed_example {
void bytes(std::initializer_list<unsigned char>);

void f() {
  // CHECK-LABEL: define{{.*}} void @_ZN13embed_example1fEv(
  // CHECK: alloca [2 x i8],
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 1 {{.*}}, ptr align 1 @.str, i64 2,
  // CHECK: call void @_ZN13embed_example5bytesESt16initializer_listIhE(
  bytes({
#embed <jk.txt>
  });
}
} // namespace embed_example

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
