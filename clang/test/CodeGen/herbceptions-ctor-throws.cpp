// RUN: %clang_cc1 -std=c++20 -fherbceptions -emit-llvm -o - %s | FileCheck %s

// A `throws` constructor must carry the error discriminant through its
// (normally void) return slot, exactly like an ordinary throws function.
// Without the herbception ABI on constructors/destructors,
// arrangeCXXStructorDeclaration left CurFnInfo without HasThrowsReturn, so
// StartFunction set ReturnValue to an invalid Address. EmitHerbceptionTry
// then called DataLayout::getTypeAllocSize on a null Type* and crashed codegen.

namespace std {
struct error {
  void *domain;
  __SIZE_TYPE__ code;
  ~error() noexcept;
};
} // namespace std

int open() throws;

struct file {
  int fd;
  // The member-initializer calls a throws function; Sema wraps it in
  // `try(open())`, which requires the enclosing constructor to have the
  // throws return ABI.
  file() throws : fd(open()) {}
};

// A throws constructor called from a throws function auto-propagates: Sema
// does not wrap CXXConstructExpr in `try()`, so EmitCall must take the error
// path itself.
int use() throws {
  file f{};
  return f.fd;
}

// The enclosing throws function uses the {E, i1} ABI (emitted before the ctor).
// CHECK: define dso_local { { ptr, i64 }, i1 } @_Z3usev()

// The constructor definition uses the {E, i1} ABI, not void.
// CHECK: define {{.*}} { { ptr, i64 }, i1 } @_ZN4fileC2Ev(

// The constructor reads the discriminant of the wrapped throws call and stores
// it (true) into its own discriminant slot on the error path.
// CHECK: store i1 true, ptr %
