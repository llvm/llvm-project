// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fconstexpr-steps=1024 -emit-llvm -o - %s | FileCheck %s

// Reaching -fconstexpr-steps says the evaluator gave up on the array bound, not
// that the bound is erroneous. [expr.new] only lets a *new-expression* whose
// array bound is erroneous evaluate to a null pointer, so an allocation whose
// bound we merely could not evaluate has to happen at run time and its result
// has to be used. Clang used to fold the pointer to null while still emitting
// the call, so every later read saw nullptr and the block leaked.
// https://github.com/llvm/llvm-project/issues/246281

namespace std {
using size_t = decltype(sizeof(0));
struct nothrow_t {};
inline constexpr nothrow_t nothrow{};
} // namespace std

void *operator new[](std::size_t, const std::nothrow_t &) noexcept;
void operator delete[](void *) noexcept;

// One element more than -fconstexpr-steps allows: representable, so not
// erroneous, so not a constant either.
int overLimit() {
  const int n = 1025;
  unsigned char *const p = new (std::nothrow) unsigned char[n];
  if (p) {
    operator delete[](p);
    return 1;
  }
  return 0;
}

// The condition is compared against the value the call returned, rather than
// folded, and the block that is freed is the one that was loaded back.
// CHECK-LABEL: define{{.*}} @_Z9overLimitv
// CHECK:      %[[ALLOC:.+]] = call{{.*}} ptr @_ZnamRKSt9nothrow_t
// CHECK:      store ptr %[[ALLOC]]
// CHECK:      icmp ne ptr
// CHECK:      %[[FREED:.+]] = load ptr, ptr
// CHECK:      call void @_ZdaPv(ptr {{.*}}%[[FREED]])

// A bound no address space can hold is an erroneous new-expression, so the
// non-throwing allocation still evaluates to a null pointer.
int tooBig() {
  unsigned char *const p = new (std::nothrow) unsigned char[4294967296ULL];
  if (p) {
    operator delete[](p);
    return 1;
  }
  return 0;
}

// CHECK-LABEL: define{{.*}} @_Z6tooBigv
// CHECK-NOT:   icmp
// CHECK:       ret i32 0
