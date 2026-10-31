// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -disable-llvm-passes -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -disable-llvm-passes -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -disable-llvm-passes -fclangir -emit-cir %s -o %t0.cir
// RUN: FileCheck --input-file=%t0.cir %s -check-prefix=CIR0
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -disable-llvm-passes -fclangir -emit-llvm %s -o %t0.ll
// RUN: FileCheck --input-file=%t0.ll %s -check-prefix=LLVM0
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O0 -disable-llvm-passes -emit-llvm %s -o %t0.ll
// RUN: FileCheck --input-file=%t0.ll %s -check-prefix=LLVM0

__attribute__((cold)) void cold_decl(void);

__attribute__((cold)) void cold_fn(void) {}
// CIR: cir.func{{.*}}@cold_fn()
// CIR-SAME: cold
// CIR-SAME: optsize
// LLVM: define{{.*}}@cold_fn(){{.*}} #[[COLD_ATTR:[0-9]+]] {
// CIR0: cir.func{{.*}}@cold_fn()
// CIR0-NOT: optsize
// LLVM0: define{{.*}}@cold_fn(){{.*}} #[[COLD_ATTR:[0-9]+]] {

void plain_fn(void) {}
// CIR: cir.func{{.*}}@plain_fn()
// CIR-NOT: optsize
// LLVM: define{{.*}}@plain_fn(){{.*}} #[[PLAIN_ATTR:[0-9]+]] {
// CIR0: cir.func{{.*}}@plain_fn()
// CIR0-NOT: optsize
// LLVM0: define{{.*}}@plain_fn(){{.*}} #[[PLAIN_ATTR:[0-9]+]] {

// optnone suppresses optsize.
__attribute__((cold)) __attribute__((optnone)) void cold_optnone_fn(void) {}
// CIR: cir.func{{.*}}@cold_optnone_fn()
// CIR-NOT: optsize
// LLVM: define{{.*}}@cold_optnone_fn(){{.*}} #[[OPTNONE_ATTR:[0-9]+]] {
// CIR0: cir.func{{.*}}@cold_optnone_fn()
// CIR0-NOT: optsize
// At O0 this shares cold_fn's group (both get O0 optnone); its content is
// pinned by the COLD check below.
// LLVM0: define{{.*}}@cold_optnone_fn(){{.*}} #[[OPTNONE_ATTR:[0-9]+]] {

// Minsize keeps optsize on even at O0, by suppressing the O0-implies-optnone
// default.
__attribute__((cold)) __attribute__((minsize)) void cold_minsize_fn(void) {}
// CIR: cir.func{{.*}}@cold_minsize_fn()
// CIR-SAME: cold
// CIR-SAME: minsize
// CIR-SAME: optsize
// LLVM: define{{.*}}@cold_minsize_fn(){{.*}} #[[MINSIZE_ATTR:[0-9]+]] {
// CIR0: cir.func{{.*}}@cold_minsize_fn()
// CIR0-SAME: cold
// CIR0-SAME: minsize
// CIR0-SAME: optsize
// LLVM0: define{{.*}}@cold_minsize_fn(){{.*}} #[[MINSIZE_ATTR:[0-9]+]] {

void caller_fn(void) {
  cold_fn();
  cold_decl();
}
// CIR: cir.func{{.*}}@caller_fn()
// CIR: cir.call{{.*}}@cold_fn()
// CIR-NOT: optsize
// CIR: cir.call{{.*}}@cold_decl()
// CIR-NOT: optsize
// CIR: cir.return
// LLVM: define{{.*}}@caller_fn()
// LLVM: call void @cold_fn() #[[CALL_ATTR:[0-9]+]]
// LLVM: call void @cold_decl()

// Declarations emit last in both pipelines.
// CIR: cir.func{{.*}}@cold_decl()
// Delimited: the trailing #loc lines carry this file's own name.
// CIR-NOT: optsize,
// CIR-NOT: optsize}
// LLVM: declare{{.*}}@cold_decl(){{.*}} #[[DECL_ATTR:[0-9]+]]

// LLVM: attributes #[[COLD_ATTR]] = {
// LLVM-SAME: cold
// LLVM-SAME: optsize
// LLVM: attributes #[[PLAIN_ATTR]] = {
// LLVM-NOT: optsize
// LLVM: attributes #[[OPTNONE_ATTR]] = {
// LLVM-SAME: optnone
// LLVM-NOT: optsize
// LLVM: attributes #[[MINSIZE_ATTR]] = {
// LLVM-SAME: minsize
// LLVM-SAME: optsize
// LLVM: attributes #[[DECL_ATTR]] = {
// LLVM-NOT: optsize
// LLVM: attributes #[[CALL_ATTR]] = { cold }

// LLVM0: attributes #[[COLD_ATTR]] = {
// LLVM0-SAME: optnone
// LLVM0-NOT: optsize
// LLVM0: attributes #[[PLAIN_ATTR]] = {
// LLVM0-NOT: optsize
// LLVM0: attributes #[[MINSIZE_ATTR]] = {
// LLVM0-SAME: minsize
// LLVM0-SAME: optsize
