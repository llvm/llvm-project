// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -o - %s                                         | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux -emit-llvm -o - %s -fexperimental-new-constant-interpreter | FileCheck %s

template<typename T>
struct Wrapper {
  T *Val;

  template<typename _Up>
  constexpr Wrapper(_Up&& __u) {
    T& __f = static_cast<_Up&&>(__u);
    Val = &__f;
  }
  constexpr T& get() const { return *Val; }
};

void f(){}
int main() {
  auto W = Wrapper<decltype(f)>(f);

  if (&W.get() != &f)
    __builtin_abort();
}

/// We used to convert do the pointer->fnptr conversion
/// by doing an integer conversion in between, which caused the
/// %0 line to be:
/// store ptr inttoptr (i64 138574454870464 to ptr), ptr %__f, align 8
// CHECK: @_ZN7WrapperIFvvEEC2IRS0_EEOT_
// CHECK: %0 = load ptr, ptr %__u.addr
