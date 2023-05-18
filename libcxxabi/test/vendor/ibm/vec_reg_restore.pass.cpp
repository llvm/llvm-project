//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that the PowerPC vector registers are restored properly during
// unwinding. Option -mabi=vec-extabi is required to compile the test case.

// REQUIRES: target=powerpc{{(64)?}}-ibm-aix
// ADDITIONAL_COMPILE_FLAGS: -mabi=vec-extabi
// UNSUPPORTED: no-exceptions

// AIX does not support the eh_frame section. Instead, the traceback table
// located at the end of each function provides the information for stack
// unwinding. Non-volatile GRs, FRs, and VRs clobbered by the function are
// saved on the stack and the numbers of saved registers are available in the
// traceback table. Registers are saved from high number to low consecutively,
// e.g., if n VRs are saved, the order on the stack will be VR31, VR30, ...,
// VR31-n+1. This test cases checks the unwinder gets to the location of saved
// VRs which should be 16-byte aligned and restores them correctly based on
// the number specified in the traceback table. To simplify, only the 2 high
// numbered VRs are checked. Because PowerPC CPUs do not have instructions to
// assign a literal value to a VR directly until Power10, and the instructions
// to assign to a VR from a GR and vice versa are not available until Power8,
// vector instructions available on Power7 are used to facilitate the test
// so that it can run on all supported PowerPC architectures. In the code
// below, VR31 is equivalent to VS63, VR30 is equivalent to VS62 (see PowerPC
// documents for details).
//

#include <cstdlib>
#include <cassert>

int __attribute__((noinline)) test2(int i)
{
  if (i > 3)
    throw i;
  srand(i);
  return rand();
}

int __attribute__((noinline)) test(int i) {
  // Clobber VS63 and VS62 in the function body.
  // Set VS63 to 16 bytes each with value 9
  asm volatile("vspltisb 31, 9" : : : "v31");

  // Set VS62 to 16 bytes each with value 12
  asm volatile("vspltisb 30, 12" : : : "v30");
  return test2(i);
}
#define cmpVS63(vec, result)                                                                                           \
  {                                                                                                                    \
    vector unsigned char gbg;                                                                                          \
    asm volatile("vcmpequb. %[gbg], 31, %[veca];"                                                                      \
                 "mfocrf %[res], 2;"                                                                                   \
                 "rlwinm %[res], %[res], 25, 31, 31"                                                                   \
                 : [res] "=r"(result), [gbg] "=v"(gbg)                                                                 \
                 : [veca] "v"(vec)                                                                                     \
                 : "cr6");                                                                                             \
  }

#define cmpVS62(vec, result)                                                                                           \
  {                                                                                                                    \
    vector unsigned char gbg;                                                                                          \
    asm volatile("vcmpequb. %[gbg], 30, %[veca];"                                                                      \
                 "mfocrf %[res], 2;"                                                                                   \
                 "rlwinm %[res], %[res], 25, 31, 31"                                                                   \
                 : [res] "=r"(result), [gbg] "=v"(gbg)                                                                 \
                 : [veca] "v"(vec)                                                                                     \
                 : "cr6");                                                                                             \
  }
int main(int, char**) {
  // Set VS63 to 16 bytes each with value 1
  asm volatile("vspltisb 31, 1" : : : "v31");

  // Set VS62 to 16 bytes each with value 2
  asm volatile("vspltisb 30, 2" : : : "v30");
  vector unsigned long long expectedVS63Value = {0x101010101010101, 0x101010101010101};
  vector unsigned long long expectedVS62Value = {0x202020202020202, 0x202020202020202};
  try {
    test(4);
  } catch (int num) {
    // If the unwinder restores VS63 and VS62 correctly, they should contain
    // 0x01's and 0x02's respectively instead of 0x09's and 0x12's.
    bool isEqualVS63, isEqualVS62;
    cmpVS63(expectedVS63Value, isEqualVS63);
    cmpVS62(expectedVS62Value, isEqualVS62);
    assert(isEqualVS63 && isEqualVS62);
  }
  return 0;
}
