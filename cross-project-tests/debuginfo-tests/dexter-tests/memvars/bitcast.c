// XFAIL:*
//// Suboptimal coverage, see description below.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %clang -std=gnu11 -O3 -glldb %s -o %t
// RUN: %dexter --fail-lt 1.0 -w --debugger lldb --binary %t -- %s

//// Adapted from https://bugs.llvm.org/show_bug.cgi?id=34136#c1
//// LowerDbgDeclare has since been updated to look through bitcasts. We still
//// get suboptimal coverage at the beginning of 'main' though. For each local,
//// LowerDbgDeclare inserts a dbg.value and a dbg.value+DW_OP_deref before the
//// store (after the call to 'getint') and the call to 'alias' respectively.
//// The first dbg.value describes the result of the 'getint' call, eventually
//// becoming a register location. The second points back into the stack
//// home. There is a gap in the coverage between the quickly clobbered register
//// location and the stack location, even though the stack location is valid
//// during that gap. For x86 we end up with this code at the start of main:
//// 00000000004004b0 <main>:
////   4004b0:  sub    rsp,0x18
////   4004b4:  mov    edi,0x5
////   4004b9:  call   400480 <getint>
////   4004be:  mov    DWORD PTR [rsp+0x14],eax
////   4004c2:  mov    edi,0x5
////   4004c7:  call   400480 <getint>
////   4004cc:  mov    DWORD PTR [rsp+0x10],eax
////   4004d0:  mov    edi,0x5
////   4004d5:  call   400480 <getint>
////   4004da:  mov    DWORD PTR [rsp+0xc],eax
////   ...
//// With these variable locations:
////  DW_TAG_variable
////    DW_AT_location        (0x00000000:
////       [0x00000000004004be, 0x00000000004004cc): DW_OP_reg0 RAX
////       [0x00000000004004de, 0x0000000000400503): DW_OP_breg7 RSP+20)
////    DW_AT_name    ("x")
////    ...
////  DW_TAG_variable
////    DW_AT_location        (0x00000037:
////       [0x00000000004004cc, 0x00000000004004da): DW_OP_reg0 RAX
////       [0x00000000004004e8, 0x0000000000400503): DW_OP_breg7 RSP+16)
////    DW_AT_name    ("y")
////    ...
////  DW_TAG_variable
////    DW_AT_location        (0x0000006e:
////       [0x00000000004004da, 0x00000000004004e8): DW_OP_reg0 RAX
////       [0x00000000004004f2, 0x0000000000400503): DW_OP_breg7 RSP+12)
////    DW_AT_name    ("z")
////    ...

char g = 1;
int five = 5;
__attribute__((__noinline__))
int getint(int x) {
  g = x - 4;
  return x * g;
}

__attribute__((__noinline__))
void alias(char* c) {
  g = *c;
  *c = (char)five;
}

int main() {
  int x = getint(5);
  int y = getint(5); // DexLabel('s1')
  int z = getint(5); // DexLabel('s2')
  alias((char*)&x);  // DexLabel('s3')
  alias((char*)&y);
  alias((char*)&z);
  return 0;          // DexLabel('s4')
}

// DexExpectWatchValue('x', '5',  from_line=ref('s1'), to_line=ref('s4'))
// DexExpectWatchValue('y', '5',  from_line=ref('s2'), to_line=ref('s4'))
// DexExpectWatchValue('z', '5',  from_line=ref('s3'), to_line=ref('s4'))
