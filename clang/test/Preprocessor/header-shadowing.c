// RUN: rm -rf %t
// RUN: split-file %s %t

/// Check that:
/// - Quoted includes ("...") trigger the diagnostic.
/// - System headers are ignored.
/// - #include_next does not cause a duplicate warning.
// RUN: %clang_cc1 -Wshadow-header -Eonly %t/main.c -I %t/include1 -I %t/include2 \
// RUN: -isystem %t/system1 -isystem %t/system2 2>&1 | FileCheck %s --check-prefix=SHADOWING

// SHADOWING: {{.*}} warning: multiple candidates for header 'header.h' found; directory '{{.*}}include1' chosen, ignoring others including '{{.*}}include2' [-Wshadow-header]
// SHADOWING: warning: include1/header.h included!
// SHADOWING-NOT: {{.*}} warning: multiple candidates for header 'header.h' found; directory '{{.*}}include2' chosen, ignoring others including '{{.*}}include1' [-Wshadow-header]
// SHADOWING: warning: include2/header.h included!
// SHADOWING-NOT: {{.*}} warning: multiple candidates for header 'stdio.h' found; directory '{{.*}}system1' chosen, ignoring others including '{{.*}}system2' [-Wshadow-header]
// SHADOWING: warning: system1/stdio.h included!

/// Check that the diagnostic is only performed once in MSVC header search mode.
// RUN: %clang_cc1 -fheader-search=microsoft -Wshadow-header -Eonly %t/t.c 2>&1 | FileCheck %s --check-prefix=SHADOWING-MS

// SHADOWING-MS: {{.*}} warning: multiple candidates for header 't3.h' found; directory '{{.*}}foo' chosen, ignoring others including '{{.*}}' [-Wshadow-header]
// SHADOWING-MS-NOT: {{.*}} warning: multiple candidates for header 't3.h' found; directory '{{.*}}' chosen, ignoring others including '{{.*}}foo' [-Wshadow-header]
// SHADOWING-MS: warning: Found foo/t3.h.

//--- main.c
#include "header.h"
#include <stdio.h>

//--- include1/header.h
#warning include1/header.h included!
#include_next "header.h"

//--- include2/header.h
#warning include2/header.h included!

//--- system1/stdio.h
#warning system1/stdio.h included!

//--- system2/stdio.h
#warning system2/stdio.h included!


/// Used to test when running in MSVC compatibility
//--- t.c
#include "foo/t1.h"

//--- foo/t1.h
#include "bar/t2.h"

//--- foo/bar/t2.h
#include "t3.h"

//--- foo/t3.h
#warning Found foo/t3.h.

//--- t3.h
#warning Found t3.h.
