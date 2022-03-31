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
// e.g., if n VRs are saved, the order on the stack will be VR63, VR62, ...,
// VR63-n+1. This test cases checks the unwinder gets to the location of saved
// VRs which should be 16-byte aligned and restores them correctly based on
// the number specified in the traceback table. To simplify, only the 2 high
// numbered VRs are checked. Because PowerPC CPUs do not have instructions to
// assign a literal value to a VR directly until Power10, the value is
// assigned to a GR and then from the GR to a VR in the code.
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
  // Clobber VR63 and VR62 in the function body.
  // Set VR63=100.
  asm volatile("li 30, 100\n\t"
               "mtvsrd 63, 30\n\t"
               :
               :
               :  "v31", "r30");
  // Set VR62=200.
  asm volatile("li 29, 200\n\t"
               "mtvsrd 62, 29\n\t"
               :
               :
               :  "v30", "r29");
  return test2(i);
}

// Return the value of VR63 in 'output'.
#define getFirstValue(output) \
   asm volatile( "mfvsrd 4, 63\n\t" \
                 "std 4, %[d]" \
          : [d] "=rm"(output) \
          : \
          : )
// Return the value of VR62 in 'output'.
#define getSecondValue(output) \
   asm volatile( "mfvsrd 4, 62\n\t" \
                 "std 4, %[d]" \
          : [d] "=rm"(output) \
          : \
          : )

int main(int, char**) {
  // Set VR63=1.
  asm volatile("li 30, 1\n\t"
               "mtvsrd 63, 30\n\t"
               :
               :
               :  "v31", "r30");
  // Set VR62=1.
  asm volatile("li 29, 2\n\t"
               "mtvsrd 62, 29\n\t"
               :
               :
               :  "v30", "r29");
  long long old;
  long long old2;
  getFirstValue(old);
  getSecondValue(old2);
  try {
    test(4);
  } catch (int num) {
    long long new_value;
    long long new_value2;
    getFirstValue(new_value);
    getSecondValue(new_value2);
    // If the unwinder restores VR63 and VR62 correctly, they should contain
    // 1 and 2 respectively instead of 100 and 200.
    assert(old == new_value && old2 == new_value2);
  }
  return 0;
}
