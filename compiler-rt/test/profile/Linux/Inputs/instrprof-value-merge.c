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

// CHECK:      Counters:
// CHECK-NEXT:   main:
// CHECK-NEXT:     Hash: 0x0a9bd81e87ab6e87
// CHECK-NEXT:     Counters: 6
// CHECK-NEXT:     Indirect Call Site Count: 3
// CHECK-NEXT:     Number of Memory Intrinsics Calls: 3
// CHECK-NEXT:     Block counts: [27, 297, 12, 132, 3, 2]
// CHECK-NEXT:     Indirect Target Results:
// CHECK-NEXT:         [ 0, foo, 3 ]
// CHECK-NEXT:         [ 1, foo, 27 ]
// CHECK-NEXT:         [ 2, foo, 297 ]
// CHECK-NEXT:     Memory Intrinsic Size Results:
// CHECK-NEXT:         [ 0, 4, 2 ]
// CHECK-NEXT:         [ 1, 5, 12 ]
// CHECK-NEXT:         [ 2, 6, 132 ]
// CHECK-NEXT: Instrumentation level: IR  entry_first = 0
// CHECK-NEXT: Functions shown: 1
// CHECK-NEXT: Total functions: 3
// CHECK-NEXT: Maximum function count: 327
// CHECK-NEXT: Maximum internal block count: 297
// CHECK-NEXT: Total number of blocks: 8
// CHECK-NEXT: Total count: 805
// CHECK-NEXT: Statistics for indirect call sites profile:
// CHECK-NEXT:   Total number of sites: 3
// CHECK-NEXT:   Total number of sites with values: 3
// CHECK-NEXT:   Total number of profiled values: 3
// CHECK-NEXT:   Value sites histogram:
// CHECK-NEXT:         NumTargets, SiteCount
// CHECK-NEXT:         1, 3
// CHECK-NEXT: Statistics for memory intrinsic calls sizes profile:
// CHECK-NEXT:   Total number of sites: 3
// CHECK-NEXT:   Total number of sites with values: 3
// CHECK-NEXT:   Total number of profiled values: 3
// CHECK-NEXT:   Value sites histogram:
// CHECK-NEXT:         NumTargets, SiteCount
// CHECK-NEXT:         1, 3
