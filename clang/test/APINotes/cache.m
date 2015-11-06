// RUN: rm -rf %t/APINotesCache
// RUN: %clang_cc1 -fapinotes -fapinotes-cache-path=%t/APINotesCache -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -verify

// Check for the presence of the cached compiled form.
// RUN: ls %t/APINotesCache | grep "APINotes-.*.apinotesc"
// RUN: ls %t/APINotesCache | grep "SomeKit-.*.apinotesc"

// Run test again to ensure that caching doesn't cause problems.
// RUN: %clang_cc1 -fapinotes -fapinotes-cache-path=%t/APINotesCache -I %S/Inputs/Headers -F %S/Inputs/Frameworks  %s -verify

// Check that the driver provides a default -fapinotes-cache-path=
// RUN: %clang -fsyntax-only -fapinotes -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -### 2>&1 | FileCheck --check-prefix=CHECK-DEFAULT-PATH %s
// CHECK-DEFAULT-PATH: -fapinotes-cache-path={{.*}}org.llvm.clang/APINotesCache

// Check that the driver passes through a provided -fapinotes-cache-path=
// RUN: %clang -fsyntax-only -fapinotes -fapinotes-cache-path=/wobble -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -### 2>&1 | FileCheck --check-prefix=CHECK-PATH %s
// CHECK-PATH: -fapinotes-cache-path=/wobble

#include "HeaderLib.h"
#import <SomeKit/SomeKit.h>

int main() {
  int i;
  i = unavailable_function(); // expected-error{{'unavailable_function' is unavailable: I beg you not to use this}}
  // expected-note@HeaderLib.h:8{{'unavailable_function' has been explicitly marked unavailable here}}

  A *a = 0;
  [a transform:a]; // expected-error{{'transform:' is unavailable: anything but this}}
  // expected-note@SomeKit/SomeKit.h:6{{'transform:' has been explicitly marked unavailable here}}

  return 0;
}

