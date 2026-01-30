// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

extern "C" {
  // CIR: cir.func{{.*}}@returns_twice() attributes {returns_twice} {
  // LLVM: Function Attrs:{{.*}}returns_twice
  // LLVM-NEXT: define{{.*}}@returns_twice() #[[RT_ATTR:.*]] {
  __attribute__((returns_twice))
  void returns_twice() {}
  // CIR: cir.func{{.*}}@cold() attributes {cold} {
  // LLVM: Function Attrs:{{.*}}cold
  // LLVM-NEXT: define{{.*}}@cold() #[[COLD_ATTR:.*]] {
  __attribute__((cold))
  void cold() {}
  // CIR: cir.func{{.*}}@hot() attributes {hot} {
  // LLVM: Function Attrs:{{.*}}hot
  // LLVM-NEXT: define{{.*}}@hot() #[[HOT_ATTR:.*]] {
  __attribute__((hot))
  void hot() {}
  // CIR: cir.func{{.*}}@nodupes() attributes {noduplicate} {
  // LLVM: Function Attrs:{{.*}}noduplicate
  // LLVM-NEXT: define{{.*}}@nodupes() #[[ND_ATTR:.*]] {
  __attribute__((noduplicate))
  void nodupes() {}
  // CIR: cir.func{{.*}}@convergent() attributes {convergent} {
  // LLVM: Function Attrs:{{.*}}convergent
  // LLVM-NEXT: define{{.*}}@convergent() #[[CONV_ATTR:.*]] {
  __attribute__((convergent))
  void convergent() {}

  void caller() {
  // CIR: cir.call @returns_twice() {returns_twice} : () -> ()
  // LLVM: call void @returns_twice() #[[RT_CALL_ATTR:.*]]
    returns_twice();
  // CIR: cir.call @cold() {cold} : () -> ()
  // LLVM: call void @cold() #[[COLD_CALL_ATTR:.*]]
    cold();
  // CIR: cir.call @hot() {hot} : () -> ()
  // LLVM: call void @hot() #[[HOT_CALL_ATTR:.*]]
    hot();
  // CIR: cir.call @nodupes() {noduplicate} : () -> ()
  // LLVM: call void @nodupes() #[[ND_CALL_ATTR:.*]]
    nodupes();
  // CIR: cir.call @convergent() {convergent} : () -> ()
  // LLVM: call void @convergent() #[[CONV_CALL_ATTR:.*]]
    convergent();
  }
}

// LLVM: attributes #[[RT_ATTR]] = {{.*}}returns_twice
// LLVM: attributes #[[COLD_ATTR]] = {{.*}}cold
// LLVM: attributes #[[HOT_ATTR]] = {{.*}}hot
// LLVM: attributes #[[ND_ATTR]] = {{.*}}noduplicate
// LLVM: attributes #[[CONV_ATTR]] = {{.*}}convergent
// LLVM: attributes #[[RT_CALL_ATTR]] = {{.*}}returns_twice
// LLVM: attributes #[[COLD_CALL_ATTR]] = {{.*}}cold
// LLVM: attributes #[[HOT_CALL_ATTR]] = {{.*}}hot
// LLVM: attributes #[[ND_CALL_ATTR]] = {{.*}}noduplicate
// LLVM: attributes #[[CONV_CALL_ATTR]] = {{.*}}convergent
