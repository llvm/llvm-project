// RUN: %clang_tsan -O1 %s -o %t && %run %t 2>&1 | FileCheck %s
// REQUIRES: x86_64-target-arch
// UNSUPPORTED: darwin
#include "test.h"

struct ucontext {
  void *sp;
  void *fiber;
};

extern "C" {
  void ucontext_do_switch(void **save, void **load);
  void ucontext_trampoline();
}

__asm__(".global ucontext_do_switch\n"
        "ucontext_do_switch:\n\t"
        "pushq %rbp\n\t"
        "pushq %r15\n\t"
        "pushq %r14\n\t"
        "pushq %r13\n\t"
        "pushq %r12\n\t"
        "pushq %rbx\n\t"
        "movq %rsp, (%rdi)\n\t"
        "movq (%rsi), %rsp\n\t"
        "popq %rbx\n\t"
        "popq %r12\n\t"
        "popq %r13\n\t"
        "popq %r14\n\t"
        "popq %r15\n\t"
        "popq %rbp\n\t"
        "retq");

__asm__(".global ucontext_trampoline\n"
        "ucontext_trampoline:\n\t"
        ".cfi_startproc\n\t"
        ".cfi_undefined rip\n\t"
        "movq %r12, %rdi\n\t"
        "jmpq *%rbx\n\t"
        ".cfi_endproc");

void ucontext_init(ucontext *context, void *stack, unsigned stack_sz,
                   void (*func)(void*), void *arg) {
  void **sp = reinterpret_cast<void **>(static_cast<char *>(stack) + stack_sz);
  *(--sp) = 0;
  *(--sp) = reinterpret_cast<void *>(ucontext_trampoline);
  *(--sp) = 0;   // rbp
  *(--sp) = 0;   // r15
  *(--sp) = 0;   // r14
  *(--sp) = 0;   // r13
  *(--sp) = arg; // r12
  *(--sp) = reinterpret_cast<void *>(func); // rbx
  context->sp = sp;
  context->fiber = __tsan_create_fiber(0);
}

void ucontext_free(ucontext *context) {
  __tsan_destroy_fiber(context->fiber);
}

__attribute__((no_sanitize_thread))
void ucontext_switch(ucontext *save, ucontext *load) {
  save->fiber = __tsan_get_current_fiber();
  __tsan_switch_to_fiber(load->fiber, 0);
  ucontext_do_switch(&save->sp, &load->sp);
}

char stack[64 * 1024] __attribute__((aligned(16)));

ucontext uc, orig_uc;

void func(void *arg) {
  __asm__ __volatile__(".cfi_undefined rip");
  ucontext_switch(&uc, &orig_uc);
}

int main() {
  ucontext_init(&uc, stack, sizeof(stack), func, 0);
  ucontext_switch(&orig_uc, &uc);
  ucontext_free(&uc);
  fprintf(stderr, "PASS\n");
  return 0;
}

// CHECK-NOT: WARNING: ThreadSanitizer:
// CHECK: PASS
