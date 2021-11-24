// RUN: %clangxx %s -fsanitize-memory-track-origins=1 -o %t
// RUN: %env_tool_opts="compress_stack_depot=0:malloc_context_size=128:verbosity=1" %run %t 2>&1 | FileCheck %s --implicit-check-not="StackDepot released"
// RUN: %env_tool_opts="compress_stack_depot=1:malloc_context_size=128:verbosity=1" %run %t 2>&1 | FileCheck %s --check-prefixes=COMPRESS

// Ubsan does not store stacks.
// UNSUPPORTED: ubsan

// FIXME: Fails for unknown reason.
// UNSUPPORTED: s390x

#include <memory>

__attribute__((noinline)) void a(unsigned v);
__attribute__((noinline)) void b(unsigned v);

std::unique_ptr<int[]> p;

__attribute__((noinline)) void a(unsigned v) {
  int x;
  v >>= 1;
  if (!v) {
    p.reset(new int[100]);
    p[1] = x;
    return;
  }
  if (v & 1)
    b(v);
  else
    a(v);
}

__attribute__((noinline)) void b(unsigned v) { return a(v); }

int main(int argc, char *argv[]) {
  for (unsigned i = 0; i < 100000; ++i)
    a(i + (i << 16));
  return 0;
}

// COMPRESS: StackDepot released {{[0-9]+}}
