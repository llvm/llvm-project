// This file checks output given when processing C/ObjC files.
// When user selects invalid language standard
// print out supported values with short description.

// RUN: not %clang %s -std=foobar -c 2>&1 | FileCheck --match-full-lines %s
// RUN: not %clang -x objective-c %s -std=foobar -c 2>&1 | FileCheck --match-full-lines %s
// RUN: not %clang -x renderscript %s -std=foobar -c 2>&1 | FileCheck --match-full-lines %s

// CHECK: error: invalid value 'foobar' in '-std=foobar'
// CHECK-NEXT: note: use 'c89', 'c90', or 'iso9899:1990' for 'ISO C 1990' standard
// CHECK-NEXT: note: use 'iso9899:199409' for 'ISO C 1990 with amendment 1' standard
// CHECK-NEXT: note: use 'gnu89' or 'gnu90' for 'ISO C 1990 with GNU extensions' standard
// CHECK-NEXT: note: use 'c99' or 'iso9899:1999' for 'ISO C 1999' standard
// CHECK-NEXT: note: use 'gnu99' for 'ISO C 1999 with GNU extensions' standard
// CHECK-NEXT: note: use 'c11' or 'iso9899:2011' for 'ISO C 2011' standard
// CHECK-NEXT: note: use 'gnu11' for 'ISO C 2011 with GNU extensions' standard
// CHECK-NEXT: note: use 'c17', 'iso9899:2017', 'c18', or 'iso9899:2018' for 'ISO C 2017' standard
// CHECK-NEXT: note: use 'gnu17' or 'gnu18' for 'ISO C 2017 with GNU extensions' standard
// CHECK-NEXT: note: use 'c23' for 'Working Draft for ISO C23' standard
// CHECK-NEXT: note: use 'gnu23' for 'Working Draft for ISO C23 with GNU extensions' standard

// Make sure that no other output is present.
// CHECK-NOT: {{^.+$}}

