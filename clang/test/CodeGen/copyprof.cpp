// Test that CopyProf instrumentation passes are invoked at -O0 and -O2,
// and are not re-run during ThinLTO postlink backend compilation.

// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -fcopyprof %s -fdebug-pass-manager -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix=INSTRUMENT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fcopyprof %s -fdebug-pass-manager -emit-llvm -o /dev/null 2>&1 | FileCheck %s --check-prefix=INSTRUMENT

// INSTRUMENT: Running pass: CopyProfPass on
// INSTRUMENT: Running pass: ModuleCopyProfPass on [module]
// INSTRUMENT: Running pass: CopyProfStoresPass on

// Test ThinLTO prelink vs postlink:
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fcopyprof -flto=thin -emit-llvm-bc %s -o %t.bc
// RUN: llvm-lto -thinlto -o %t %t.bc
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fcopyprof -fthinlto-index=%t.thinlto.bc -fdebug-pass-manager -emit-obj -x ir %t.bc -o /dev/null 2>&1 | FileCheck %s --check-prefix=POSTLINK

// POSTLINK-NOT: Running pass: CopyProfPass
// POSTLINK-NOT: Running pass: ModuleCopyProfPass
// POSTLINK-NOT: Running pass: CopyProfStoresPass

struct Foo {
  long a[2];
  Foo() : a{0, 0} {}
  Foo(const Foo &other) {
    a[0] = other.a[0];
    a[1] = other.a[1];
  }
  ~Foo() {}
};

int main() {
  Foo f1;
  Foo f2(f1);
  return 0;
}
