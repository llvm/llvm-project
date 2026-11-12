// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fsanitize=kernel-memory \
// RUN:   -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fsanitize=kernel-memory \
// RUN:   -S -o - %s | FileCheck %s --check-prefix=ASM

// The way the Linux kernel uses KMSAN instrumentation regions: the markers are
// volatile asm statements that hold nothing but a comment. They survive the
// optimization pipeline with the calls staying between them, and
// KernelMemorySanitizer passes clean parameter shadow to those calls, in a
// function it otherwise must not touch.

#define instrumentation_begin() __asm__ volatile("# kmsan.instrumentation.begin")
#define instrumentation_end() __asm__ volatile("# kmsan.instrumentation.end")
#define noinstr __attribute__((disable_sanitizer_instrumentation))

struct pt_regs {
  long ip, sp;
};
struct pair {
  long a, b;
};

void enter(struct pt_regs *regs);
void leave(struct pt_regs *regs);
void handle(struct pair p);
int printk(const char *fmt, ...);

// WARN_ON() opens a region of its own, which nests when it is used in one.
static inline __attribute__((always_inline)) void warn_on(long cond) {
  instrumentation_begin();
  if (cond)
    printk("warning\n");
  instrumentation_end();
}

// CHECK-LABEL: define {{.*}}void @entry(
// CHECK-NOT:     __msan
// CHECK:         call void @enter(
// CHECK-NEXT:    call void asm sideeffect "# kmsan.instrumentation.begin"
// CHECK:         call void asm sideeffect "# kmsan.instrumentation.begin"
//
// The printk() in warn_on() has no variadic arguments in the overflow area:
// CHECK:         [[CTX1:%.*]] = call ptr @__msan_get_context_state()
// CHECK:         store i64 0, ptr
// CHECK-NEXT:    call {{.*}}@printk(
//
// CHECK:         call void asm sideeffect "# kmsan.instrumentation.end"
//
// handle() takes the struct in two registers, which are not noundef:
// CHECK-NEXT:    [[CTX2:%.*]] = call ptr @__msan_get_context_state()
// CHECK-NEXT:    store i64 0, ptr [[CTX2]]
// CHECK:         store i64 0, ptr
// CHECK-NEXT:    call void @handle(
//
// CHECK-NEXT:    call void asm sideeffect "# kmsan.instrumentation.end"
// CHECK-NEXT:    call void @leave(
// CHECK-NEXT:    ret void
noinstr void entry(struct pt_regs *regs, struct pair p) {
  enter(regs);
  instrumentation_begin();
  warn_on(regs->ip);
  handle(p);
  instrumentation_end();
  leave(regs);
}

// A function that is not noinstr is instrumented as a whole; the markers do
// not change that.
// CHECK-LABEL: define {{.*}}void @instrumented(
// CHECK:         call ptr @__msan_get_context_state()
// CHECK-NOT:     call ptr @__msan_get_context_state()
// CHECK:         ret void
void instrumented(struct pt_regs *regs, struct pair p) {
  instrumentation_begin();
  handle(p);
  instrumentation_end();
  leave(regs);
}

// The markers generate no code, and neither does the function get a KMSAN
// prologue.
// ASM-LABEL: entry:
// ASM-NOT:     __msan
// ASM:         callq enter
// ASM-NEXT:    #APP
// ASM-NEXT:    # kmsan.instrumentation.begin
// ASM-NEXT:    #NO_APP
// ASM:         #APP
// ASM-NEXT:    # kmsan.instrumentation.begin
// ASM-NEXT:    #NO_APP
// ASM:         callq __msan_get_context_state
// ASM:         callq printk
// ASM:         #APP
// ASM-NEXT:    # kmsan.instrumentation.end
// ASM-NEXT:    #NO_APP
// ASM:         callq __msan_get_context_state
// ASM:         callq handle
// ASM:         #APP
// ASM-NEXT:    # kmsan.instrumentation.end
// ASM-NEXT:    #NO_APP
// ASM-NOT:     __msan
// ASM:         jmp leave
