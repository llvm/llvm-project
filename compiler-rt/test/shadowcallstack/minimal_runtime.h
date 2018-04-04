// A shadow call stack runtime is not yet included with compiler-rt, provide a
// minimal runtime to allocate a shadow call stack and assign %gs to point at
// it.

#pragma once

#include <asm/prctl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/prctl.h>

int arch_prctl(int code, void *addr);

__attribute__((no_sanitize("shadow-call-stack")))
static void __shadowcallstack_init() {
  void *stack = mmap(NULL, 8192, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (stack == MAP_FAILED)
    abort();

  if (arch_prctl(ARCH_SET_GS, stack))
    abort();
}

__attribute__((section(".preinit_array"), used))
    void (*__shadowcallstack_preinit)(void) = __shadowcallstack_init;
