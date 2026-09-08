// RUN: %clang -std=c++26 -fherbceptions -fno-exceptions -S -emit-llvm -o - %s | FileCheck %s

// A constructor that throws a herbception inside a `try { } catch throws(...)`
// block must route the error to the catch handler. Previously the switch in
// the continuation block had two cases for i32 0 instead of i32 0 and i32 1,
// which made the catch handler unreachable when the first constructor throws.

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

// Struct that throws on construction
struct bad_file {
  int fd;
  bad_file() throws {
    throw throws ::std::errc::io_error;
  }
  ~bad_file() noexcept {}
};

// Struct that succeeds on construction
struct good_file {
  int fd;
  good_file() throws : fd(1) {}
  ~good_file() noexcept {}
};

// When the first constructor throws, the catch handler must be reachable.
// CHECK: define dso_local noundef i32 @_Z8test_bugv()
// CHECK: call { { ptr, i64 }, i1 } @_ZN8bad_fileC2Ev
// CHECK: extractvalue { { ptr, i64 }, i1 } %{{.*}}, 1
// CHECK: br i1 %{{.*}}, label %{{[0-9]+}}, label %{{[0-9]+}}
// CHECK: store %"struct.std::error" %{{.*}}, ptr %
// CHECK: switch i32 %{{.*}}, label %{{[0-9]+}} [
// CHECK: i32 {{[0-9]+}}, label %{{[0-9]+}}
// CHECK: i32 {{[0-9]+}}, label %{{[0-9]+}}
// CHECK-NEXT: ]

int test_bug() try {
  bad_file f1;      // Throws
  good_file f2;     // Should not reach
  return 0;
} catch throws(std::error e) {
  return 1;
}
