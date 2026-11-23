// RUN: clang-tidy -checks='-*,clang-diagnostic-invalid-offsetof,concurrency-mt-unsafe' -header-filter='.*' -system-headers=true %s -- -isystem %S/Inputs/system-headers 2>&1 | FileCheck -check-prefix=CHECK-SYSTEM-HEADERS %s
// RUN: clang-tidy -checks='-*,clang-diagnostic-invalid-offsetof,concurrency-mt-unsafe' -header-filter='.*' -system-headers=false %s -- -isystem %S/Inputs/system-headers 2>&1 | FileCheck -check-prefix=CHECK-NO-SYSTEM-HEADERS %s

// Validate that we don't get the diagnostics when the 'clang-diagnostic-*'
// check is disabled:
// RUN: clang-tidy -checks='-*,concurrency-mt-unsafe' -header-filter='.*' -system-headers=true %s -- -isystem %S/Inputs/system-headers 2>&1 | FileCheck -implicit-check-not='{{warning:|error:}}' %s

// FIXME: The check 'concurrency-mt-unsafe' is completely unrelated to this
// test, it is only added to the RUN lines because Clang-Tidy aborts the
// analysis with "Error: no checks enabled." if all the enabled checks are
// 'clang-diagnostic-*' checks (i.e. compiler warnings).
// Once GH#192713 is resolved, remove 'concurrency-mt-unsafe'.

#include <mock_cstddef.h>

struct D {
  virtual void f() {}
  virtual ~D() {}
  int i;
};

int main() {
  // Previously Clang-Tidy was suppressing this -Winvalid-offsetof report
  // because the error location is in a system macro (namely, 'offsetof').
  (void) offsetof(D, i);
  // CHECK-SYSTEM-HEADERS: :[[@LINE-1]]:10: warning: 'offsetof' on non-standard-layout type 'D' [clang-diagnostic-invalid-offsetof]
  // CHECK-NO-SYSTEM-HEADERS: :[[@LINE-2]]:10: warning: 'offsetof' on non-standard-layout type 'D' [clang-diagnostic-invalid-offsetof]
}
