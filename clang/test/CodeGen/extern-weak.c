// RUN: %clang_cc1 -O3 -emit-llvm -o - %s | grep extern_weak
// RUN: %clang_cc1 -O3 -emit-llvm -o - %s | llc

#if !defined(__linux__) && !defined(__FreeBSD__) && \
    !defined(__OpenBSD__) && !defined(__CYGWIN__) && !defined(__DragonFly__)
void foo(void) __attribute__((weak_import));
#else
void foo(void) __attribute__((weak));
#endif

void bar(void) { foo(); }

