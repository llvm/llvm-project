#include <string.h>

void (*f0)();
void (*f1)();
void (*f2)();

char dst[200];
char src[200];
volatile int n;

__attribute__((noinline)) void foo() {}

__attribute__((noinline)) void bar() {
  f0 = foo;
  f1 = foo;
  f2 = foo;
  n = 4;
}
int main(int argc, char *argv[]) {
  int i;
  bar();
  if (argc == 1) {
    f0();
    for (i = 0; i < 9; i++)
      f1();
    for (i = 0; i < 99; i++)
      f2();
  } else {
    memcpy((void *)dst, (void *)src, n);
    for (i = 0; i < 6; i++)
      memcpy((void *)(dst + 2), (void *)src, n + 1);
    for (i = 0; i < 66; i++)
      memcpy((void *)(dst + 9), (void *)src, n + 2);
  }
}

// CHECK: Counters:
// CHECK:   main:
// CHECK:     Hash: 0x0a9bd81e87ab6e87
// CHECK:     Counters: 6
// CHECK:     Indirect Call Site Count: 3
// CHECK:     Number of Memory Intrinsics Calls: 3
// CHECK:     Block counts: [27, 297, 12, 132, 3, 2]
// CHECK:     Indirect Target Results:
// CHECK:         [ 0, foo, 3 ]
// CHECK:         [ 1, foo, 27 ]
// CHECK:         [ 2, foo, 297 ]
// CHECK:     Memory Intrinsic Size Results:
// CHECK:         [ 0, 4, 2 ]
// CHECK:         [ 1, 5, 12 ]
// CHECK:         [ 2, 6, 132 ]
// CHECK: Instrumentation level: IR
// CHECK: Functions shown: 1
// CHECK: Total functions: 3
// CHECK: Maximum function count: 327
// CHECK: Maximum internal block count: 297
// CHECK: Statistics for indirect call sites profile:
// CHECK:   Total number of sites: 3
// CHECK:   Total number of sites with values: 3
// CHECK:   Total number of profiled values: 3
// CHECK:   Value sites histogram:
// CHECK:         NumTargets, SiteCount
// CHECK:         1, 3
// CHECK: Statistics for memory intrinsic calls sizes profile:
// CHECK:   Total number of sites: 3
// CHECK:   Total number of sites with values: 3
// CHECK:   Total number of profiled values: 3
// CHECK:   Value sites histogram:
// CHECK:         NumTargets, SiteCount
// CHECK:         1, 3
