// Check that DW_AT_entry_pc is correctly rewritten
// to the relocated instruction by BOLT.

// RUN: gcc -gdwarf-5 -O3 -Wl,-q %s -o %t
// RUN: llvm-bolt --update-debug-sections %t -o %t.bolt
// RUN: llvm-objdump -d %t.bolt > %t.dump
// RUN: llvm-dwarfdump %t.bolt >> %t.dump
// RUN: FileCheck %s < %t.dump

// CHECK-LABEL: <live>:
// CHECK: [[#%x,ENTRY1:]]: {{.*}} mov w0, w3
// CHECK: [[#%x,ENTRY2:]]: {{.*}} add w2, w2, #0x1

// CHECK: DW_TAG_inlined_subroutine
// CHECK-NEXT: DW_AT_abstract_origin {{.*}}"inner")
// CHECK-NEXT: DW_AT_entry_pc (0x{{0*}}[[#ENTRY1]])
// CHECK: DW_TAG_inlined_subroutine
// CHECK-NEXT: DW_AT_abstract_origin {{.*}}"inner")
// CHECK-NEXT: DW_AT_entry_pc (0x{{0*}}[[#ENTRY2]])

volatile int sink;

__attribute__((noinline)) int other(int v) { return v * 5 + 2; }

static inline __attribute__((always_inline)) int inner(int a) {
  int t = other(a);
  if (__builtin_expect(t > 100, 0))
    sink = t;
  return t + 1;
}

__attribute__((noinline)) int live(int x) {
  int s = 0;
  for (int i = 0; i < x; ++i)
    s += inner(s) + inner(i);
  return s;
}

int main(int argc, char **) {
  sink = live(argc);
  return 0;
}
