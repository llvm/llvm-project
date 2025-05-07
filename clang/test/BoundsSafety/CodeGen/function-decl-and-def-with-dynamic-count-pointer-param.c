
// This is a test to see if the compiler doesn't crash.

// RUN: %clang_cc1 -O0  -fbounds-safety -Wno-int-conversion %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -Wno-int-conversion %s -o /dev/null
// RUN: %clang_cc1 -O0  -fbounds-safety -Wno-int-conversion -x objective-c -fexperimental-bounds-safety-objc %s -o /dev/null
// RUN: %clang_cc1 -O2  -fbounds-safety -Wno-int-conversion -x objective-c -fexperimental-bounds-safety-objc %s -o /dev/null

#define a(b, ...) __builtin___memmove_chk(b, __VA_ARGS__, b)
#define c(d, h, e) a(d, h, e)
#define f(g) __attribute__((counted_by(g)))
int n;
void *j;
void l(unsigned char *f(*i), unsigned long *i);
void l(unsigned char *f(*i) k, unsigned long *i) { c(k, (char*)j, n); }
