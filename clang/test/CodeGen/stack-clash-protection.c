// check interaction between -fstack-clash-protection and dynamic allocation schemes
// RUN: %clang -target x86_64 -O0 -o %t.out %s -fstack-clash-protection && %t.out

#include <alloca.h>
#include <string.h>

int large_stack() __attribute__((noinline));

int large_stack() {
  int stack[20000], i;
  for (i = 0; i < sizeof(stack) / sizeof(int); ++i)
    stack[i] = i;
  return stack[1];
}

int main(int argc, char **argv) {
  int volatile static_mem[8000];
  for (size_t i = 0; i < argc * sizeof(static_mem) / sizeof(static_mem[0]); ++i)
    static_mem[i] = argc * i;

  int vla[argc];
  memset(&vla[0], 0, argc);

  int index = large_stack();

  // also check allocation of 0 size
  volatile void *mem = __builtin_alloca(argc - 1);

  int volatile *dyn_mem = alloca(sizeof(static_mem) * argc);
  for (size_t i = 0; i < argc * sizeof(static_mem) / sizeof(static_mem[0]); ++i)
    dyn_mem[i] = argc * i;

  return static_mem[(7999 * argc) / 2] - dyn_mem[(7999 * argc) / 2] + vla[argc - index];
}
