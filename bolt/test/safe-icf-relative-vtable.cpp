// Test safe ICF works with binaries that contain relative vtable.

// REQUIRES: system-linux,asserts

// RUN: %clang %cxxflags -o %t.so %s -Wl,-q -fno-rtti
// RUN: llvm-bolt %t.so -o %t.bolt --no-threads --icf=safe \
// RUN:   --debug-only=bolt-icf 2>&1 | FileCheck %s

// RUN: %clang %cxxflags -o %t.exp.so %s -Wl,-q -fno-rtti \
// RUN:   -fexperimental-relative-c++-abi-vtables
// RUN: llvm-bolt %t.exp.so -o %t.exp.bolt --no-threads --icf=safe \
// RUN:   --debug-only=bolt-icf 2>&1 | FileCheck %s

// CHECK: folding {{.*bar.*}} into {{.*foo.*}}
// CHECK-NOT: skipping function with reference taken {{.*bar.*}}

class TT {
public:
  virtual int foo(int a) { return ++a; }
  virtual int bar(int a) { return ++a; }
};

// __init_array_end was added to check that ICF works correctly even if the
// symbol has the same address as a vtable symbol
extern void (*__init_array_end[])();
__attribute__((used)) static void *dummy__init_array_end = __init_array_end;
__attribute__((constructor)) void dummy_ctor() {};

int main() {
  TT T;
  return T.foo(0) + T.bar(1);
}
