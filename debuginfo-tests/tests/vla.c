// This test case verifies the debug location for variable-length arrays.
// RUN: %clang %target_itanium_abi_host_triple -O0 -g %s -c -o %t.o
// RUN: %clang %target_itanium_abi_host_triple %t.o -o %t.out
// RUN: %test_debuginfo %s %t.out
//
// DEBUGGER: break 18
// DEBUGGER: r
// DEBUGGER: p vla[0]
// CHECK: 23
// DEBUGGER: p vla[1]
// CHECK: 22

void init_vla(int size) {
  int i;
  int vla[size];
  for (i = 0; i < size; i++)
    vla[i] = size-i;
  vla[0] = size; // line 18
}

int main(int argc, const char **argv) {
  init_vla(23);
  return 0;
}
