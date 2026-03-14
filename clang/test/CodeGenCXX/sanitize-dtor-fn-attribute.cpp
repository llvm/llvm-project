// Test -fsanitize-memory-use-after-dtor
// RUN: %clang_cc1 -fsanitize=memory -fsanitize-memory-use-after-dtor -std=c++11 -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s --implicit-check-not="call void @__sanitizer_"

// The no_sanitize_memory attribute, when applied to a destructor,
// represses emission of sanitizing callback

template <class T> class Vector {
 public:
  int size;
  ~Vector() {}
};

struct No_San {
  Vector<int> v;
  int x;
  No_San() { }
  __attribute__((no_sanitize_memory)) ~No_San() = default;
};

int main() {
  No_San *ns = new No_San();
  ns->~No_San();
  return 0;
}

// Repressing the sanitization attribute results in no msan
// instrumentation of the destructor
// CHECK: define {{.*}}No_SanD1Ev{{.*}} [[ATTRIBUTE:#[0-9]+]]
// CHECK: ret void

// CHECK: define {{.*}}No_SanD2Ev{{.*}} [[ATTRIBUTE:#[0-9]+]]
// CHECK: call void {{.*}}VectorIiED2Ev
// CHECK: ret void

// CHECK: define {{.*}}VectorIiED2Ev
// CHECK: call void @__sanitizer_dtor_callback
// CHECK: ret void

// When attribute is repressed, the destructor does not emit any tail calls
// CHECK-NOT: attributes [[ATTRIBUTE]] = {{.*}} sanitize_memory
