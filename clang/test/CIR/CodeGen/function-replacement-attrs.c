// RUN: %clang_cc1 -std=c11 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.cir
// RUN: %clang_cc1 -std=c11 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -disable-llvm-passes %s -o %t-cir.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t-cir.ll
// RUN: %clang_cc1 -std=c11 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck %s --check-prefix=LLVM --input-file=%t.ll

// Replacing a function declaration must preserve call-site attributes even
// when its prototype changes. The asm names make distinct AST declarations
// refer to the same CIR function, so their attributes are not inherited by Sema.

extern void *old_direct(char *, ...) __asm__("direct")
    __attribute__((const, returns_nonnull));

void *call_direct(char *p) { return old_direct(p, 1); }

// The actual argument types match the new prototype, so this remains direct.
// CIR-LABEL: cir.func{{.*}}@call_direct(
// CIR: cir.call @direct(%{{.*}}, %{{.*}}) nounwind willreturn {memory_effects = #cir.memory_effects<other = none, arg_mem = none,
// CIR-SAME: : (!cir.ptr<!s8i> {llvm.noundef}, !s32i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull})
// LLVM-LABEL: define{{.*}} ptr @call_direct(
// LLVM: call nonnull ptr @direct(ptr noundef %{{.*}}, i32 noundef 1) #[[CALL_ATTRS:[0-9]+]]

void *direct(char *p, int count) { return p; }

extern void *old_indirect(int) __asm__("indirect")
    __attribute__((const, returns_nonnull));

void *call_indirect(int value) { return old_indirect(value); }

// Signed and unsigned integers have distinct CIR types. The rewritten call
// therefore needs a pointer cast, while keeping its original call attributes.
// CIR-LABEL: cir.func{{.*}}@call_indirect(
// CIR: %[[ADDR:.*]] = cir.get_global @indirect : !cir.ptr<!cir.func<(!u32i) -> !cir.ptr<!void>>>
// CIR-NEXT: %[[CALLEE:.*]] = cir.cast bitcast %[[ADDR]] : !cir.ptr<!cir.func<(!u32i) -> !cir.ptr<!void>>> -> !cir.ptr<!cir.func<(!s32i) -> !cir.ptr<!void>>>
// CIR-NEXT: cir.call %[[CALLEE]](%{{.*}}) nounwind willreturn {memory_effects = #cir.memory_effects<other = none, arg_mem = none,
// CIR-SAME: : (!cir.ptr<!cir.func<(!s32i) -> !cir.ptr<!void>>>, !s32i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.nonnull})
// LLVM-LABEL: define{{.*}} ptr @call_indirect(
// LLVM: call nonnull ptr @indirect(i32 noundef %{{.*}}) #[[CALL_ATTRS]]

int result;
void *indirect(unsigned value) { return &result; }

// A different declaration sharing the asm name must not impose its function
// attributes on the definition. Merely taking legacy's address does not make
// calls to the prototyped replacement noreturn.
extern int legacy() __asm__("replacement") __attribute__((noreturn));
int (*legacy_address)() = legacy;

int replacement(int value) { return value; }

// CIR-LABEL: cir.func{{.*}}@replacement(
// CIR-NOT: noreturn
// CIR: cir.return
// LLVM-LABEL: define{{.*}} i32 @replacement(
// LLVM-SAME: ) #[[REPLACEMENT_ATTRS:[0-9]+]]
// LLVM: ret i32

int call_replacement(void) { return replacement(1); }

// CIR-LABEL: cir.func{{.*}}@call_replacement(
// CIR: cir.call @replacement(
// CIR: cir.return
// LLVM-LABEL: define{{.*}} i32 @call_replacement(
// LLVM: call i32 @replacement(i32 noundef 1)
// LLVM: ret i32

// LLVM-DAG: attributes #[[CALL_ATTRS]] = { nounwind willreturn memory(none) }
// LLVM-DAG: attributes #[[REPLACEMENT_ATTRS]] = { noinline nounwind optnone
