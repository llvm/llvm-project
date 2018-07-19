// Ensure that we have a quiet startup when we don't have the XRay
// instrumentation sleds.
//
// RUN: %clangxx -std=c++11 %s -o %t %xraylib
// RUN: XRAY_OPTIONS="patch_premain=true verbosity=1" %run %t 2>&1 | \
// RUN:    FileCheck %s --check-prefix NOISY
// RUN: XRAY_OPTIONS="patch_premain=true verbosity=0" %run %t 2>&1 | \
// RUN:    FileCheck %s --check-prefix QUIET
// RUN: XRAY_OPTIONS="" %run %t 2>&1 | FileCheck %s --check-prefix DEFAULT
//
// FIXME: Understand how to make this work on other platforms
// REQUIRES: built-in-llvm-tree
// REQUIRES: x86_64-target-arch
#include <iostream>

using namespace std;

int main(int, char**) {
  // NOISY: {{.*}}XRay instrumentation map missing. Not initializing XRay.
  // QUIET-NOT: {{.*}}XRay instrumentation map missing. Not initializing XRay.
  // DEFAULT-NOT: {{.*}}XRay instrumentation map missing. Not initializing XRay.
  cout << "Hello, XRay!" << endl;
  // NOISY: Hello, XRay!
  // QUIET: Hello, XRay!
  // DEFAULT: Hello, XRay!
}
