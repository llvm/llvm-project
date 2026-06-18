/* This test check that the negate-ra-state CFIs are properly emitted in case of
   function splitting. The test checks two things:
    - we split at the correct location: to test the feature,
        we need to split *before* the bl __cxa_throw@PLT call is made,
        so the unwinder has to unwind from the split (cold) part.

    - the BOLTed binary runs, and returns the string from foo.

# REQUIRES: system-linux,bolt-runtime

# FDATA: 1 main #split# 1 _Z3foov 0 0 1

# RUN: %clangxx --target=aarch64-unknown-linux-gnu \
# RUN: -mbranch-protection=pac-ret %s -o %t.exe -Wl,-q
# RUN: link_fdata %s %t.exe %t.fdata
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions --split-eh \
# RUN: --split-strategy=profile2 --split-all-cold --print-split \
# RUN: --print-only=_Z3foov --data=%t.fdata 2>&1 | FileCheck \
# RUN: --check-prefix=BOLT-CHECK %s
# RUN: %t.bolt | FileCheck %s  --check-prefix=RUN-CHECK

# BOLT-CHECK-NOT: bl      __cxa_throw@PLT
# BOLT-CHECK: -------   HOT-COLD SPLIT POINT   -------
# BOLT-CHECK: bl      __cxa_throw@PLT

# RUN-CHECK: Exception caught: Exception from foo().
*/

#include <cstdio>
#include <stdexcept>

void foo() { throw std::runtime_error("Exception from foo()."); }

int main() {
  try {
    __asm__ __volatile__("split:");
    foo();
  } catch (const std::exception &e) {
    printf("Exception caught: %s\n", e.what());
  }
  return 0;
}
