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

  // CIR: cir.func{{.*}}@no_caller_saved_registers() attributes {no_caller_saved_registers} {
  // LLVM: Function Attrs:
  // LLVM-NOT: no_caller_saved_registers
  // LLVM-NEXT: define{{.*}}@no_caller_saved_registers() #[[NCSR_ATTR:.*]] {
  __attribute__((no_caller_saved_registers))
  void no_caller_saved_registers() {}

  // CIR: cir.func{{.*}}@leaf() attributes {nocallback} {
  // LLVM: Function Attrs:
  // LLVM-NOT: leaf
  // LLVM-NEXT: define{{.*}}@leaf() #[[LEAF_ATTR:.*]] {
  __attribute__((leaf))
  void leaf() {}

  // CIR: cir.func{{.*}}@modular_format({{.*}}) attributes {modular_format = "kprintf,1,2,someIdent,someStr,aspect,aspect2"} {
  // LLVM: Function Attrs:
  // LLVM-NOT:modular_format
  // LLVM-NEXT: define{{.*}}@modular_format({{.*}}) #[[MOD_FORMAT_ATTR:.*]] {
  __attribute__((format(kprintf, 1, 2)))
  __attribute__((modular_format(someIdent, "someStr", "aspect", "aspect2")))
  void modular_format(const char *c, ...) {}

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

  // CIR: cir.call @no_caller_saved_registers() {no_caller_saved_registers} : () -> ()
  // LLVM: call void @no_caller_saved_registers() #[[NCSR_CALL_ATTR:.*]]
    no_caller_saved_registers();

  // CIR: cir.call @leaf() {nocallback} : () -> ()
  // LLVM: call void @leaf() #[[LEAF_CALL_ATTR:.*]]
    leaf();

  // CIR: cir.call @modular_format({{.*}}) {modular_format = "kprintf,1,2,someIdent,someStr,aspect,aspect2"} : 
  // LLVM: call void {{.*}}@modular_format({{.*}}) #[[MOD_FORMAT_CALL_ATTR:.*]]
    modular_format("");
  }
}

// LLVM: attributes #[[RT_ATTR]] = {{.*}}returns_twice
// LLVM: attributes #[[COLD_ATTR]] = {{.*}}cold
// LLVM: attributes #[[HOT_ATTR]] = {{.*}}hot
// LLVM: attributes #[[ND_ATTR]] = {{.*}}noduplicate
// LLVM: attributes #[[CONV_ATTR]] = {{.*}}convergent
// LLVM: attributes #[[NCSR_ATTR]] = {{.*}}no_caller_saved_registers
// LLVM: attributes #[[LEAF_ATTR]] = {{.*}}nocallback
// LLVM: attributes #[[MOD_FORMAT_ATTR]] = {{.*}}"modular-format"="kprintf,1,2,someIdent,someStr,aspect,aspect2"
// LLVM: attributes #[[RT_CALL_ATTR]] = {{.*}}returns_twice
// LLVM: attributes #[[COLD_CALL_ATTR]] = {{.*}}cold
// LLVM: attributes #[[HOT_CALL_ATTR]] = {{.*}}hot
// LLVM: attributes #[[ND_CALL_ATTR]] = {{.*}}noduplicate
// LLVM: attributes #[[CONV_CALL_ATTR]] = {{.*}}convergent
// LLVM: attributes #[[NCSR_CALL_ATTR]] = {{.*}}no_caller_saved_registers
// LLVM: attributes #[[LEAF_CALL_ATTR]] = {{.*}}nocallback
// LLVM: attributes #[[MOD_FORMAT_CALL_ATTR]] = {{.*}}"modular-format"="kprintf,1,2,someIdent,someStr,aspect,aspect2"
