// RUN: %clang %s --target=%itanium_abi_triple -S -O2 -emit-llvm -o - \
// RUN:   | FileCheck %s --check-prefix=NO-EXTENSION --implicit-check-not="call void (...) @llvm.fake.use("
// Check that we do not generate fake_use instructions without -fextend-variable-liveness=this
// or -fextend-variable-liveness=all, even with `#pragma clang extend_lifetimes enable`
// declared.

// RUN: %clang %s --target=%itanium_abi_triple -S -O2 -emit-llvm -fextend-variable-liveness=this -o - \
// RUN:   | FileCheck %s --implicit-check-not="call void (...) @llvm.fake.use("
// Check that we generate a fake_use instruction for this 'this' ptr in foo()
// and baz(), but not for bar().

// RUN: %clang %s --target=%itanium_abi_triple -S -O2 -emit-llvm -fextend-variable-liveness=all -o - \
// RUN:   | FileCheck %s --check-prefixes CHECK,ALL --implicit-check-not="call void (...) @llvm.fake.use("
// Check that we generate a fake_use instruction for all parameters (including
// the 'this' ptr) in foo() and baz(), but not for bar().

class A {
  void foo(int i);
  void bar(int i);
  void bat(int i);
  void baz(int i);
  void goo(int i);
  void moo(int i);
  void zoo(int i);
  void fie(int i);
};

// NO-EXTENSION: define{{.*}}foo
// Lifetime extension is initially enabled by the command-line option.
void A::foo(int i) {
  // CHECK-LABEL: define{{.*}}foo
  // ALL:         call void (...) @llvm.fake.use(i32 %
  // CHECK:       call void (...) @llvm.fake.use(ptr nonnull %
}

#pragma clang extend_lifetimes disable
// Disable lifetime extension for subsequent function definitions.
void A::bar(int i) {
  // CHECK-LABEL: define{{.*}}bar
  // CHECK-NOT:   fake.use
  // CHECK:       ret
}

// The disabling pragma spans multiple function definitions.
void A::bat(int i) {
  // CHECK-LABEL: define{{.*}}bat
  // CHECK-NOT:   fake.use
  // CHECK:       ret
}

#pragma clang extend_lifetimes enable
// Enable lifetime extension for subsequent function definitions.
void A::baz(int i) {
  // CHECK-LABEL: define{{.*}}baz
  // ALL:         call void (...) @llvm.fake.use(i32 %
  // CHECK:       call void (...) @llvm.fake.use(ptr nonnull %
}

// Disabling inside a function does not affect that function, only subsequent ones.
void A::goo(int i) {
#pragma clang extend_lifetimes disable
  // CHECK-LABEL: define{{.*}}goo
  // ALL:         call void (...) @llvm.fake.use(i32 %
  // CHECK:       call void (...) @llvm.fake.use(ptr nonnull %
}

// The previous disable applies here; enabling inside affects subsequent functions.
void A::moo(int i) {
#pragma clang extend_lifetimes enable
  // CHECK-LABEL: define{{.*}}moo
  // CHECK-NOT:   fake.use
  // CHECK:       ret
}

// The enable in the previous function applies to this function.
void A::zoo(int i) {
  // CHECK-LABEL: define{{.*}}zoo
  // ALL:         call void (...) @llvm.fake.use(i32 %
  // CHECK:       call void (...) @llvm.fake.use(ptr nonnull %
}

// The nodebug attribute suppresses lifetime extension.
void __attribute__((nodebug)) A::fie(int i) {
  int j = 0;
  // CHECK-LABEL: define{{.*}}fie
  // CHECK-NOT:   fake.use
}
