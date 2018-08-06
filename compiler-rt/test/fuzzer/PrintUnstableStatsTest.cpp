#include <assert.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

int x = 0;
bool skip0 = false;
bool skip1 = false;
bool skip2 = false;

__attribute__((noinline)) void det0() { x++; }
__attribute__((noinline)) void det1() { x++; }
__attribute__((noinline)) void det2() { x++; }
__attribute__((noinline)) void det3() { x++; }
__attribute__((noinline)) void det4() { x++; }

__attribute__((noinline)) void ini0() { x++; }
__attribute__((noinline)) void ini1() { x++; }
__attribute__((noinline)) void ini2() { x++; }

__attribute__((noinline)) void t0() { x++; }
__attribute__((noinline)) void t1() { x++; }
__attribute__((noinline)) void t2() { x++; }
__attribute__((noinline)) void t3() { x++; }
__attribute__((noinline)) void t4() { x++; }

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size) {
  if (Size == 1 && Data[0] == 'A' && !skip0) {
    skip0 = true;
    ini0();
  }
  if (Size == 1 && Data[0] == 'B' && !skip1) {
    skip1 = true;
    ini1();
  }
  if (Size == 1 && Data[0] == 'C' && !skip2) {
    skip2 = true;
    ini2();
  }

  det0();
  det1();
  int a = rand();
  det2();

  switch (a % 5) {
  case 0:
    t0();
    break;
  case 1:
    t1();
    break;
  case 2:
    t2();
    break;
  case 3:
    t3();
    break;
  case 4:
    t4();
    break;
  default:
    assert(false);
  }

  det3();
  det4();
  return 0;
}
