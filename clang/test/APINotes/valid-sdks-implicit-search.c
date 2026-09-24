// RUN: rm -rf %t && split-file %s %t

// On macOS 10.15 the inner notes apply, so the outer notes are not used.
// RUN: %clang_cc1 -triple arm64-apple-macosx10.15 -fapinotes \
// RUN:   -isysroot %S/../Driver/Inputs/MacOSX10.15.sdk \
// RUN:   -fsyntax-only -I %t/outer/inner %t/test.c -verify=inner

// On macOS 15.0 the inner notes are skipped and the outer notes are used.
// RUN: %clang_cc1 -triple arm64-apple-macosx15.0 -fapinotes \
// RUN:   -isysroot %S/../Driver/Inputs/MacOSX15.0.sdk \
// RUN:   -fsyntax-only -I %t/outer/inner %t/test.c -verify=outer

//--- outer/APINotes.apinotes
---
Name: NestedNotes
# No ValidSDKs, so this always applies. Only reached when the inner file does not.
Functions:
  - Name: outer_fn
    Availability: none
    AvailabilityMsg: "from the outer directory"

//--- outer/inner/APINotes.apinotes
---
Name: NestedNotes
# Ages out on the macOS 15.0 SDK, at which point the walk has to carry on upwards.
ValidSDKs:
  - Name: macosx
    ValidUntil: 15.0
Functions:
  - Name: inner_fn
    Availability: none
    AvailabilityMsg: "from the inner directory"

//--- outer/inner/nested.h
#ifndef NESTED_H
#define NESTED_H

// Annotated by outer/inner/APINotes.apinotes.
void inner_fn(void);

// Annotated by outer/APINotes.apinotes.
void outer_fn(void);

#endif

//--- test.c
#include "nested.h"

// Exactly one of the two is annotated in each case, so the other acts as a
// negative control: reading both files, or the wrong one, shows up as an
// unexpected diagnostic.
void test(void) {
  inner_fn();
  // inner-error@-1{{'inner_fn' is unavailable: from the inner directory}}
  // inner-note@nested.h:5{{'inner_fn' has been explicitly marked unavailable here}}

  outer_fn();
  // outer-error@-1{{'outer_fn' is unavailable: from the outer directory}}
  // outer-note@nested.h:8{{'outer_fn' has been explicitly marked unavailable here}}
}
