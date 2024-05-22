// XFAIL: *
//// Suboptimal coverage, see below.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %clang -std=gnu11 -O3 -glldb %s -o %t
// RUN: %dexter --fail-lt 1.0 -w --debugger lldb --binary %t -- %s

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
  g = *p;
}

__attribute__((__noinline__))
void do_thing(int x) {
  g *= x;
}

__attribute__((__noinline__))
int fun(int param) {
  do_thing(0);                        // DexLabel('s2')
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

  return g;                           // DexLabel('s3')
}

int main() {
  return fun(5);
}

// DexExpectWatchValue('*p', 5, 5, 5, 5, 5, on_line=ref('s1'))
// DexExpectWatchValue('param', 5, from_line=ref('s2'), to_line=ref('s3'))
