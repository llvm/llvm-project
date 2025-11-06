// RUN: rm -rf %t
// RUN: split-file %s %t

/// Check that:
/// - Quoted includes ("...") trigger the diagnostic.
/// - Angled includes (<...>) are ignored.
/// - #include_next does not cause a duplicate warning.
// RUN: %clang_cc1 -Wshadow-header -Eonly %t/main.c -I %t/include1 -I %t/include2 \
// RUN: -isystem %t/system1 -isystem %t/system2 2>&1 | FileCheck %s --check-prefix=SHADOWING

// SHADOWING: {{.*}} warning: multiple candidates for header 'header.h' found; directory '{{.*}}include1' chosen, ignoring '{{.*}}include2' and others [-Wshadow-header]
// SHADOWING: warning: include1/header.h included!
// SHADOWING: warning: include2/header.h included!
// SHADOWING: warning: system1/stdio.h included!
// SHADOWING-NOT: {{.*}} warning: multiple candidates for header 'header.h' found; directory '{{.*}}include2' chosen, ignoring '{{.*}}include1' and others [-Wshadow-header]
// SHADOWING-NOT: {{.*}} warning: multiple candidates for header 'stdio.h' found; directory '{{.*}}system1' chosen, ignoring '{{.*}}system2' and others [-Wshadow-header]

/// Check that the diagnostic is only performed once in MSVC compatibility mode.
// RUN: %clang_cc1 -fms-compatibility -Wshadow-header -Eonly %t/t.c -I %t -I %t/foo -I %t/foo/bar 2>&1 | FileCheck %s --check-prefix=SHADOWING-MS

// SHADOWING-MS: {{.*}} warning: multiple candidates for header 't3.h' found; directory '{{.*}}foo' chosen, ignoring '{{.*}}' and others [-Wshadow-header]
// SHADOWING-MS: warning: Found foo/t3.h.
// SHADOWING-MS-NOT: {{.*}} warning: multiple candidates for header 't3.h' found; directory '{{.*}}foo' chosen, ignoring '{{.*}}' and others [-Wshadow-header]

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
