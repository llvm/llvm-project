// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefixes=CIR,CIR-DEF
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,LLVM-DEF
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,LLVM-DEF
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -fno-builtin-memcmp %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefixes=CIR,CIR-SPC
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fno-builtin-memcmp %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,LLVM-SPC
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fno-builtin-memcmp %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,LLVM-SPC
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -fno-builtin-memcmp -fno-builtin-memset %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefixes=CIR,CIR-BTH
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fno-builtin-memcmp -fno-builtin-memset %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,LLVM-BTH
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fno-builtin-memcmp -fno-builtin-memset %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,LLVM-BTH
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -fno-builtin %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefixes=CIR,CIR-ALL
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -fno-builtin %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,LLVM-ALL
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fno-builtin %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=LLVM,LLVM-ALL

extern "C" {
// CIR: cir.func{{.*}}@normal() attributes {
// CIR-DEF-NOT: nobuiltins
// CIR-SPC-SAME: nobuiltins = ["memcmp"]
// CIR-BTH-SAME: nobuiltins = ["memcmp", "memset"]
// CIR-ALL-SAME: nobuiltins = []
// LLVM: define{{.*}}normal() #[[NORM_ATTR:.*]] {
__attribute__((cold)) // to force attributes on the call to be around.
void normal(){}

// CIR: cir.func{{.*}}@no_builtins() attributes {
// CIR-DEF-SAME: nobuiltins = []
// CIR-SPC-SAME: nobuiltins = []
// CIR-BTH-SAME: nobuiltins = []
// CIR-ALL-SAME: nobuiltins = []
// LLVM: define{{.*}}no_builtins() #[[NB_ATTR:.*]] {
__attribute__((no_builtin))
__attribute__((hot)) // force unique attributes
void no_builtins() {}

// CIR: cir.func{{.*}}@no_memcpy() attributes {
// CIR-DEF-SAME: nobuiltins = ["memcpy"]
// CIR-SPC-SAME: nobuiltins = ["memcmp", "memcpy"]
// CIR-BTH-SAME: nobuiltins = ["memcmp", "memset", "memcpy"]
// CIR-ALL-SAME: nobuiltins = []
// LLVM: define{{.*}}no_memcpy() #[[NO_MCPY_ATTR:.*]] {
__attribute__((no_builtin("memcpy")))
__attribute__((leaf)) // force unique attributes
void no_memcpy() {}

// CIR: cir.func{{.*}}@no_memcmp() attributes {
// CIR-DEF-SAME: nobuiltins = ["memcmp"]
// CIR-SPC-SAME: nobuiltins = ["memcmp"]
// CIR-BTH-SAME: nobuiltins = ["memcmp", "memset"]
// CIR-ALL-SAME: nobuiltins = []
// LLVM: define{{.*}}no_memcmp() #[[NO_MCMP_ATTR:.*]] {
__attribute__((no_builtin("memcmp")))
__attribute__((noduplicate)) // force unique attributes
void no_memcmp() {}

// CIR: cir.func{{.*}}@no_both() attributes {
// CIR-DEF-SAME: nobuiltins = ["memcmp", "memcpy"]
// CIR-SPC-SAME: nobuiltins = ["memcmp", "memcpy"]
// CIR-BTH-SAME: nobuiltins = ["memcmp", "memset", "memcpy"]
// CIR-ALL-SAME: nobuiltins = []
// LLVM: define{{.*}}no_both() #[[NO_BOTH_ATTR:.*]] {
__attribute__((no_builtin("memcpy")))
__attribute__((no_builtin("memcmp")))
__attribute__((convergent)) // force unique attributes
void no_both(){}
}

void caller() {
  // CIR: cir.call @normal() {
  // CIR-DEF-NOT: nobuiltins
  // CIR-SPC-SAME: nobuiltins = ["memcmp"]
  // CIR-BTH-SAME: nobuiltins = ["memcmp", "memset"]
  // CIR-ALL-SAME: nobuiltins = []
  // LLVM: call void @normal() #[[NORM_CALL_ATTR:.*]]
  normal();
  // CIR: cir.call @no_builtins() {
  // CIR-DEF-SAME: nobuiltins = []
  // CIR-SPC-SAME: nobuiltins = []
  // CIR-BTH-SAME: nobuiltins = []
  // CIR-ALL-SAME: nobuiltins = []
  // LLVM: call void @no_builtins() #[[NB_CALL_ATTR:.*]]
  no_builtins();
  // CIR: cir.call @no_memcpy() {
  // CIR-DEF-SAME: nobuiltins = ["memcpy"]
  // CIR-SPC-SAME: nobuiltins = ["memcmp", "memcpy"]
  // CIR-BTH-SAME: nobuiltins = ["memcmp", "memset", "memcpy"]
  // CIR-ALL-SAME: nobuiltins = []
  // LLVM: call void @no_memcpy() #[[NO_MCPY_CALL_ATTR:.*]]
  no_memcpy();
  // CIR: cir.call @no_memcmp() {
  // CIR-DEF-SAME: nobuiltins = ["memcmp"]
  // CIR-SPC-SAME: nobuiltins = ["memcmp"]
  // CIR-BTH-SAME: nobuiltins = ["memcmp", "memset"]
  // CIR-ALL-SAME: nobuiltins = []
  // LLVM: call void @no_memcmp() #[[NO_MCMP_CALL_ATTR:.*]]
  no_memcmp();
  // CIR: cir.call @no_both() {
  // CIR-DEF-SAME: nobuiltins = ["memcmp", "memcpy"]
  // CIR-SPC-SAME: nobuiltins = ["memcmp", "memcpy"]
  // CIR-BTH-SAME: nobuiltins = ["memcmp", "memset", "memcpy"]
  // CIR-ALL-SAME: nobuiltins = []
  // LLVM: call void @no_both() #[[NO_BOTH_CALL_ATTR:.*]]
  no_both();
}

// LLVM: attributes #[[NORM_ATTR]] = {
// LLVM-DEF-NOT: no-builtin
// LLVM-SPC-SAME: "no-builtin-memcmp"
// LLVM-BTH-SAME: "no-builtin-memcmp"
// LLVM-BTH-SAME: "no-builtin-memset"
// LLVM-ALL-SAME:"no-builtins"
//
// LLVM: attributes #[[NB_ATTR]] = {
// LLVM-DEF-SAME:"no-builtins"
// LLVM-SPC-SAME:"no-builtins"
// LLVM-BTH-SAME:"no-builtins"
// LLVM-ALL-SAME:"no-builtins"
// 
// LLVM: attributes #[[NO_MCPY_ATTR]] = {
// LLVM-DEF-SAME: "no-builtin-memcpy"
// LLVM-SPC-SAME: "no-builtin-memcmp"
// LLVM-SPC-SAME: "no-builtin-memcpy"
// LLVM-BTH-SAME: "no-builtin-memcmp"
// LLVM-BTH-SAME: "no-builtin-memcpy"
// LLVM-BTH-SAME: "no-builtin-memset"
// LLVM-ALL-SAME:"no-builtins"
//
// LLVM: attributes #[[NO_MCMP_ATTR]] = {
// LLVM-DEF-SAME: "no-builtin-memcmp"
// LLVM-SPC-SAME: "no-builtin-memcmp"
// LLVM-BTH-SAME: "no-builtin-memcmp"
// LLVM-BTH-SAME: "no-builtin-memset"
// LLVM-ALL-SAME:"no-builtins"
//
// LLVM: attributes #[[NO_BOTH_ATTR]] = {
// LLVM-DEF-SAME: "no-builtin-memcmp"
// LLVM-DEF-SAME: "no-builtin-memcpy"
// LLVM-SPC-SAME: "no-builtin-memcmp"
// LLVM-SPC-SAME: "no-builtin-memcpy"
// LLVM-BTH-SAME: "no-builtin-memcmp"
// LLVM-BTH-SAME: "no-builtin-memcpy"
// LLVM-BTH-SAME: "no-builtin-memset"
// LLVM-ALL-SAME:"no-builtins"
//
//
// LLVM: attributes #[[NORM_CALL_ATTR]] = {
// LLVM-DEF-NOT: no-builtin
// LLVM-SPC-SAME: "no-builtin-memcmp"
// LLVM-BTH-SAME: "no-builtin-memcmp"
// LLVM-BTH-SAME: "no-builtin-memset"
// LLVM-ALL-SAME:"no-builtins"
//
// LLVM: attributes #[[NB_CALL_ATTR]] = {
// LLVM-DEF-SAME:"no-builtins"
// LLVM-SPC-SAME:"no-builtins"
// LLVM-BTH-SAME:"no-builtins"
// LLVM-ALL-SAME:"no-builtins"
//
// LLVM: attributes #[[NO_MCPY_CALL_ATTR]] = {
// LLVM-DEF-SAME: "no-builtin-memcpy"
// LLVM-SPC-SAME: "no-builtin-memcmp"
// LLVM-SPC-SAME: "no-builtin-memcpy"
// LLVM-BTH-SAME: "no-builtin-memcmp"
// LLVM-BTH-SAME: "no-builtin-memcpy"
// LLVM-BTH-SAME: "no-builtin-memset"
// LLVM-ALL-SAME:"no-builtins"
//
// LLVM: attributes #[[NO_MCMP_CALL_ATTR]] = {
// LLVM-DEF-SAME: "no-builtin-memcmp"
// LLVM-SPC-SAME: "no-builtin-memcmp"
// LLVM-BTH-SAME: "no-builtin-memcmp"
// LLVM-BTH-SAME: "no-builtin-memset"
// LLVM-ALL-SAME:"no-builtins"
//
// LLVM: attributes #[[NO_BOTH_CALL_ATTR]] = {
// LLVM-DEF-SAME: "no-builtin-memcmp"
// LLVM-DEF-SAME: "no-builtin-memcpy"
// LLVM-SPC-SAME: "no-builtin-memcmp"
// LLVM-SPC-SAME: "no-builtin-memcpy"
// LLVM-BTH-SAME: "no-builtin-memcmp"
// LLVM-BTH-SAME: "no-builtin-memcpy"
// LLVM-BTH-SAME: "no-builtin-memset"
// LLVM-ALL-SAME:"no-builtins"
