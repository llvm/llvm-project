// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

extern void start(void) __asm__("start");
extern void _start(void) __asm__("_start");
extern void __start(void) __asm__("__start");
void start(void) {}
void _start(void) {}
void __start(void) {}

