// RUN: %clang_cc1 -std=c++20 -fherbceptions -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-X64
// RUN: %clang_cc1 -std=c++20 -fherbceptions -emit-llvm -triple i686-linux-gnu -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-X86

// A void throws function calling a throws function that returns a struct
// of the same machine layout as the enclosing function's return slot
// used to lower the success value to the slot via
// `bitcast %struct.X to { ptr, iN }`. Under opaque pointers that bitcast
// is invalid IR (the two aggregate types share a machine layout but
// differ structurally) and was rejected by the IR verifier, crashing
// codegen during X86 DAG->DAG instruction selection in
// SelectionDAG::FoldConstantArithmetic.
//
// The success-path store in EmitHerbceptionTry must coerce through
// memory instead.

namespace std {
struct error {
  void *domain;
  __SIZE_TYPE__ code;
  ~error() noexcept;
};
template <class T> class error_domain;
template <> class error_domain<int> {
public:
  static void *domain() noexcept;
  static __SIZE_TYPE__ code(int) noexcept;
};
} // namespace std

// Match the size of std::error on each target:
//   x86_64:  void* (8) + size_t (8) = 16 bytes;  inner uses {i64, i64}
//   i686:    void* (4) + size_t (4) =  8 bytes;  inner uses {i32, i32}
namespace ns {
#ifdef __SIZEOF_POINTER__
#if __SIZEOF_POINTER__ == 8
struct status { long long p, n; };
#else
struct status { int p, n; };
#endif
#else
struct status { long long p, n; };
#endif
}

__attribute__((noinline)) ns::status inner() throws {
  ns::status s;
  asm volatile ("" : "=r"(s.p), "=r"(s.n));
  return s;
}

__attribute__((noinline)) void outer() throws {
  for (;;) {
    ns::status ret = inner();
    if (ret.p == 0) return;
  }
}

// CHECK-NOT: bitcast {{.*}} to { ptr, i
// CHECK-X64: define dso_local { { ptr, i64 }, i1 } @_Z5outerv()
// CHECK-X64: ret { { ptr, i64 }, i1 }
// CHECK-X86: define dso_local { { ptr, i32 }, i1 } @_Z5outerv()
// CHECK-X86: ret { { ptr, i32 }, i1 }
