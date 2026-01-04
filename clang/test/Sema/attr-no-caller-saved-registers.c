// RUN: %clang_cc1 %s -verify -fsyntax-only -triple powerpc-eabi

__attribute__((no_caller_saved_registers)) void valid(void) { }

#ifdef __powerpc__
__attribute__((no_caller_saved_registers)) int invalid1(void) { // expected-error{{cannot save regs if function returns value as the return value gets clobbered during restore}}
    return 0;
}

__attribute__((no_caller_saved_registers)) void* invalid2(void) { // expected-error{{cannot save regs if function returns value as the return value gets clobbered during restore}}
    return 0;
}
#endif
