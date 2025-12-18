// Test safe ICF works with binaries that contain relative vtable.

// REQUIRES: system-linux,asserts

// RUN: %clang %cxxflags -o %t.so %s -Wl,-q -fno-rtti
// RUN: llvm-bolt %t.so -o %t.bolt --no-threads --icf=safe \
// RUN:   --debug-only=bolt-icf 2>&1 | FileCheck %s

// RUN: %clang %cxxflags -o %t.so %s -Wl,-q -fno-rtti \
// RUN:   -fexperimental-relative-c++-abi-vtables
// RUN: llvm-bolt %t.so -o %t.bolt --no-threads --icf=safe \
// RUN:   --debug-only=bolt-icf 2>&1 | FileCheck %s

// CHECK: folding {{.*bar.*}} into {{.*foo.*}}
// CHECK-NOT: skipping function with reference taken {{.*bar.*}}

class TT {
public:
  virtual int foo(int a) { return ++a; }
  virtual int bar(int a) { return ++a; }
};

int main() {
  TT T;
  return T.foo(0) + T.bar(1);
}
