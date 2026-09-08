// RUN: %clang -std=c++20 -fherbceptions -fno-exceptions -S -emit-llvm -o - %s | FileCheck %s
// RUN: not %clang -std=c++20 -S -emit-llvm %s 2>&1 | FileCheck %s --check-prefix=DISABLED

// Herbception (throws): a function declared 'throws' is lowered to a {T, i1}
// return type with the llvm 'throws' attribute. The payload slot is sized to
// hold the larger of the success value T and the implicit error type
// std::error (a 2-register {void*, size_t} struct hardcoded in CodeGen), so a
// function returning int throws -> {{ptr, i64}, i1}.
// -fherbceptions is required for the keyword; it is independent of
// -fno-exceptions.

namespace std {
struct error_domain_singleton {};
struct error {
  void *d;
  __SIZE_TYPE__ c;
};
struct my_errc { int v; };
template <class T> class error_domain;
template <> class error_domain<my_errc> {
public:
  static inline constexpr error_domain_singleton const *domain() noexcept;
  static inline __SIZE_TYPE__ code(my_errc) noexcept { return 0; }
};
}

// CHECK: define dso_local { { ptr, i64 }, i1 } @_Z3fooi(i32 noundef %0) #[[ATTR:[0-9]+]]
// CHECK-NOT: call void @__cxa_throw
// CHECK: ret { { ptr, i64 }, i1 }
int foo(int x) throws {
  if (x < 0) throw throws std::my_errc{x};
  return x + 1;
}

// Plain functions (no throws) are unchanged.
// CHECK-LABEL: define dso_local noundef i32 @_Z4plinii(i32 noundef %0, i32 noundef %1)
int plin(int a, int b) { return a + b; }

// try(expr) auto-propagates the error of a throws call. The caller extracts
// the discriminant, branches on it, and on error returns {err, true}.
// CHECK-LABEL: define dso_local { { ptr, i64 }, i1 } @_Z6calleri(i32 noundef %0)
// CHECK:         call { { ptr, i64 }, i1 } @_Z3fooi
// CHECK:         extractvalue { { ptr, i64 }, i1 } %{{.*}}, 1
// CHECK:         br i1 %{{.*}}, label %{{[0-9]+}}, label %{{[0-9]+}}
int caller(int x) throws {
  return try(foo(x));
}

// CHECK: attributes #[[ATTR]] = { {{.*}}throws{{.*}} }

// Without -fherbceptions, 'throws' is not a keyword.
// DISABLED: error: expected function body after function declarator
