// RUN: rm -rf %t
// RUN: %clang_cc1 -fapinotes -fapinotes-modules -fapinotes-cache-path=%t/APINotesCache -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -verify

// Check for the presence of the cached compiled form.
// RUN: ls %t/APINotesCache | grep "APINotes-.*.apinotesc"
// RUN: ls %t/APINotesCache | grep "SomeKit-.*.apinotesc"

// Run test again to ensure that caching doesn't cause problems.
// RUN: %clang_cc1 -fapinotes -fapinotes-modules -fapinotes-cache-path=%t/APINotesCache -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -verify

// Check that the default path is taken from -fmodules-cache-path.
// RUN: %clang_cc1 -fapinotes -fapinotes-modules -fmodules-cache-path=%t/ModuleCache -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -verify
// RUN: ls %t/ModuleCache | grep "APINotes-.*.apinotesc"
// RUN: ls %t/ModuleCache | grep "SomeKit-.*.apinotesc"

// RUN: not %clang_cc1 -fapinotes -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s 2>&1 | FileCheck --check-prefix=CHECK-NO-CACHE %s
// CHECK-NO-CACHE: error: -fapinotes was provided without -fmodules-cache-path

// Check that the driver does not provide a default -fapinotes-cache-path=.
// RUN: %clang -fsyntax-only -fapinotes -fapinotes-modules -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -### 2>&1 | FileCheck --check-prefix=CHECK-DEFAULT-PATH %s
// CHECK-DEFAULT-PATH-NOT: -fapinotes-cache-path

// Check that the driver passes through a provided -fapinotes-cache-path=
// RUN: %clang -fsyntax-only -fapinotes -fapinotes-modules -fapinotes-cache-path=/wobble -I %S/Inputs/Headers -F %S/Inputs/Frameworks %s -### 2>&1 | FileCheck --check-prefix=CHECK-PATH %s
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
