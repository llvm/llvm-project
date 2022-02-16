// UNSUPPORTED: -zos, -aix
// RUN: %clang_cc1 -emit-llvm %s -o %t

typedef const struct __CFString * CFStringRef;

#define CFSTR(x) (CFStringRef) __builtin___CFStringMakeConstantString (x)

void f(void) {
  CFSTR("Hello, World!");
}

// rdar://6151192
void *G = CFSTR("yo joe");

