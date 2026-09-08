// RUN: %clang_cc1 -std=c++26 -fherbceptions -emit-llvm -o - %s | FileCheck %s

// A bare call to a `throws` function inside a `try { } catch throws(...)`
// block that is itself inside a `throws` function must route the error to the
// enclosing catch handler, not auto-propagate it to the function's return
// slot. Previously Sema auto-wrapped the bare call in a CXXTryExpr and
// EmitHerbceptionTry always auto-propagated (stored to the return slot and
// returned), which made the catch handler unreachable. The error path of the
// auto-propagation must branch to the handler when one is active.

namespace std {
struct error {
  void *domain;
  __SIZE_TYPE__ code;
  ~error() noexcept;
};
enum class errc : unsigned { io_error = 5 };
template <typename T> class error_domain;
template <> class error_domain<errc> {
public:
  static void *domain() noexcept;
  static __SIZE_TYPE__ code(errc) noexcept;
};
} // namespace std

// bar() throws a herbception (no legacy EH involved).
void bar() throws {
  throw throws ::std::errc::io_error;
}

// The catch handler must be reachable (have a predecessor) and run on the
// error path.
//
// CHECK: define dso_local { { ptr, i64 }, i1 } @_Z8test_barv()
// CHECK: call { { ptr, i64 }, i1 } @_Z3barv()
// CHECK: br i1 %{{.*}}, label %try.err, label %try.ok
// CHECK: try.err:
// CHECK: br label %catch.throws
// CHECK: catch.throws:
// CHECK-SAME: ; preds = %try.err
// CHECK: store %"struct.std::error" %{{.*}}, ptr %e, align 8
// CHECK: br label %herb.try.cont

void test_bar() throws try {
  bar();
} catch throws(std::error e) {
  (void)e;
}
