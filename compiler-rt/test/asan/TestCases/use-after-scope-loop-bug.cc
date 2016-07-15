// RUN: %clangxx_asan -O1 -fsanitize-address-use-after-scope %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s
//
// FIXME: @llvm.lifetime.* are not emitted for x.
// XFAIL: *

int *p;

int main() {
  // Variable goes in and out of scope.
  for (int i = 0; i < 3; ++i) {
    int x[3] = {i, i, i};
    p = x + i;
  }
  return *p;  // BOOM
}
