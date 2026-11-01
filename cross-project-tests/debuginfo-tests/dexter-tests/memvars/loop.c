// XFAIL: *
//// Suboptimal coverage, see below.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %clang -std=gnu11 -O3 -glldb %s -o %t
// RUN: %dexter -w %dexter_lldb_args --binary %t -- %s | FileCheck %s

//// Check that escaped local 'param' in function 'fun' has sensible debug info
//// after the escaping function 'use' gets arg promotion (int* -> int). Currently
//// we lose track of param after the loop header.

int g = 0;
//// A no-inline, read-only function with internal linkage is a good candidate
//// for arg promotion.
__attribute__((__noinline__))
static void use(const int* p) {
  //// Promoted args would be a good candidate for an DW_OP_implicit_pointer.
  //// This desirable behaviour is checked for in the test implicit-ptr.c.
  g = *p; // !dex_label s1
}

__attribute__((__noinline__))
void do_thing(int x) {
  g *= x;
}

__attribute__((__noinline__))
int fun(int param) {
  do_thing(0); // !dex_label s2
  for (int i = 0; i < param; ++i) {
    use(&param);
  }

  //// x86 loop body looks like this, with param in ebx:
  //// 4004b0: mov    edi,ebx
  //// 4004b2: call   4004d0 <_ZL3usePKi>
  //// 4004b7: add    ebp,0xffffffff
  //// 4004ba: jne    4004b0 <_Z3funi+0x20>

  //// But we lose track of param's location before the loop:
  //// DW_TAG_formal_parameter
  //// DW_AT_location   (0x00000039:
  ////    [0x0000000000400490, 0x0000000000400495): DW_OP_reg5 RDI
  ////    [0x0000000000400495, 0x00000000004004a2): DW_OP_reg3 RBX)
  //// DW_AT_name       ("param")

  return g; // !dex_label s3
}

int main() {
  return fun(5);
}

// CHECK-DAG: seen_values: 2
// CHECK-DAG: correct_step_coverage: 100.0%

/*
---
!where {lines: !label s1}:
  !value p:
    "*": 5
!where {lines: !range [!label s2, !label s3]}:
  !value param: 5
...
*/
