// Test that pruning of header search paths emulates MSVC behavior when in
// Microsoft compatibility mode. See header-search-duplicates.c for GCC
// compatible behavior.

// This test is intended to be usable to validate MSVC behavior using a .bat
// test driver similar to the following. A failure to compile successfully
// would indicate a problem with the test or, perhaps, a behavioral difference
// across MSVC versions.
//   @echo on
//   setlocal
//   rd /s/q test-msvc
//   split-file microsoft-header-search-duplicates.c test-msvc
//   pushd test-msvc
//   REM Validate test 1:
//   set INCLUDE=...
//   set EXTERNAL_INCLUDE=...
//   set EXTRA_INCLUDE=...
//   cl.exe /c /showIncludes ... test1.c
//   popd

// This test exercises both the clang and clang-cl drivers using a target that
// implicitly enables the '-fms-compatibility' option. The '-nobuiltininc',
// '-nostdinc', and '/X ('-nostdlibinc') options are used to suppress implicit
// header search paths to ease testing.

// Header search paths are processed as follows:
// 1) Paths specified by the '/I' and '/external:I' options are processed in
//    order.
//    1.1) Paths specified by '/I' that duplicate a path specified by
//         '/external:I' are ignored regardless of the option order.
//    1.2) Paths specified by '/I' that duplicate a prior '/I' option are
//         ignored.
//    1.3) Paths specified by '/external:I' that duplicate a later
//         '/external:I' option are ignored.
// 2) Paths specified by the '/external:env' options are processed in order.
//    Paths that duplicate a path from step 1, a prior '/external:env' option,
//    or a prior path from the current '/external:env' option are ignored.
// 3) Paths specified by the 'INCLUDE' environment variable are processed in
//    order. Paths that duplicate a path from step 1, step 2, or an earlier
//    path in the 'INCLUDE' environment variable are ignored.
// 4) Paths specified by the 'EXTERNALINCLUDE' environment variable are
//    processed in order. Paths that duplicate a path from step 1, step 2,
//    step 3, or an earlier path in the 'EXTERNAL_INCLUDE' environment
//    variable are ignored.

// RUN: rm -rf %t
// RUN: split-file %s %t


// Test 1: Validate ordering and duplicate elimination for /I.
//
// RUN: %clang \
// RUN:     -target x86_64-pc-windows -v -fsyntax-only \
// RUN:     -nostdinc \
// RUN:     -I%t/test1/include/y \
// RUN:     -I%t/test1/include/z \
// RUN:     -I%t/test1/include/y \
// RUN:     %t/test1/t.c 2>&1 | FileCheck -DPWD=%t %t/test1/t.c
// RUN: %clang_cl \
// RUN:     -target x86_64-pc-windows -v -fsyntax-only \
// RUN:     -nobuiltininc /X \
// RUN:     /I%t/test1/include/y \
// RUN:     /I%t/test1/include/z \
// RUN:     /I%t/test1/include/y \
// RUN:     %t/test1/t.c 2>&1 | FileCheck -DPWD=%t %t/test1/t.c

#--- test1/t.c
#include <a.h>
#include <b.h>

// CHECK:      ignoring duplicate directory "[[PWD]]/test1/include/y"
// CHECK-NEXT: #include "..." search starts here:
// CHECK-NEXT: #include <...> search starts here:
// CHECK-NEXT: [[PWD]]/test1/include/y
// CHECK-NEXT: [[PWD]]/test1/include/z
// CHECK-NEXT: End of search list.

#--- test1/include/y/a.h

#--- test1/include/z/a.h
#error 'test1/include/z/a.h' should not have been included!

#--- test1/include/z/b.h


// Test 2: Validate ordering and duplicate elimination for /external:I.
//
// RUN: %clang \
// RUN:     -target x86_64-pc-windows -v -fsyntax-only \
// RUN:     -nostdinc \
// RUN:     -iexternal %t/test2/include/z \
// RUN:     -iexternal %t/test2/include/y \
// RUN:     -iexternal %t/test2/include/z \
// RUN:     %t/test2/t.c 2>&1 | FileCheck -DPWD=%t %t/test2/t.c
// RUN: %clang_cl \
// RUN:     -target x86_64-pc-windows -v -fsyntax-only \
// RUN:     -nobuiltininc /X \
// RUN:     /external:I %t/test2/include/z \
// RUN:     /external:I %t/test2/include/y \
// RUN:     /external:I %t/test2/include/z \
// RUN:     %t/test2/t.c 2>&1 | FileCheck -DPWD=%t %t/test2/t.c

#--- test2/t.c
#include <a.h>
#include <b.h>

// CHECK:      ignoring duplicate directory "[[PWD]]/test2/include/z"
// CHECK-NEXT: #include "..." search starts here:
// CHECK-NEXT: #include <...> search starts here:
// CHECK-NEXT: [[PWD]]/test2/include/y
// CHECK-NEXT: [[PWD]]/test2/include/z
// CHECK-NEXT: End of search list.

#--- test2/include/y/a.h

#--- test2/include/z/a.h
#error 'test2/include/z/a.h' should not have been included!

#--- test2/include/z/b.h


// Test 3: Validate ordering and duplicate elimination for /I vs /external:I.
//
// RUN: %clang \
// RUN:     -target x86_64-pc-windows -v -fsyntax-only \
// RUN:     -nostdinc \
// RUN:     -iexternal %t/test3/include/w \
// RUN:     -I%t/test3/include/z \
// RUN:     -I%t/test3/include/x \
// RUN:     -I%t/test3/include/w \
// RUN:     -iexternal %t/test3/include/y \
// RUN:     -iexternal %t/test3/include/z \
// RUN:     %t/test3/t.c 2>&1 | FileCheck -DPWD=%t %t/test3/t.c
// RUN: %clang_cl \
// RUN:     -target x86_64-pc-windows -v -fsyntax-only \
// RUN:     -nobuiltininc /X \
// RUN:     /external:I %t/test3/include/w \
// RUN:     /I%t/test3/include/z \
// RUN:     /I%t/test3/include/x \
// RUN:     /I%t/test3/include/w \
// RUN:     /external:I %t/test3/include/y \
// RUN:     /external:I %t/test3/include/z \
// RUN:     %t/test3/t.c 2>&1 | FileCheck -DPWD=%t %t/test3/t.c

#--- test3/t.c
#include <a.h>
#include <b.h>
#include <c.h>
#include <d.h>

// CHECK:      ignoring duplicate directory "[[PWD]]/test3/include/w"
// CHECK-NEXT:  as it is a non-system directory that duplicates a system directory
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/test3/include/z"
// CHECK-NEXT:  as it is a non-system directory that duplicates a system directory
// CHECK-NEXT: #include "..." search starts here:
// CHECK-NEXT: #include <...> search starts here:
// CHECK-NEXT: [[PWD]]/test3/include/w
// CHECK-NEXT: [[PWD]]/test3/include/x
// CHECK-NEXT: [[PWD]]/test3/include/y
// CHECK-NEXT: [[PWD]]/test3/include/z
// CHECK-NEXT: End of search list.

#--- test3/include/w/a.h

#--- test3/include/x/a.h
#error 'test3/include/x/a.h' should not have been included!

#--- test3/include/x/b.h

#--- test3/include/y/a.h
#error 'test3/include/y/a.h' should not have been included!

#--- test3/include/y/b.h
#error 'test3/include/y/b.h' should not have been included!

#--- test3/include/y/c.h

#--- test3/include/z/a.h
#error 'test3/include/z/a.h' should not have been included!

#--- test3/include/z/b.h
#error 'test3/include/z/b.h' should not have been included!

#--- test3/include/z/c.h
#error 'test3/include/z/c.h' should not have been included!

#--- test3/include/z/d.h


// Test 4: Validate ordering and duplicate elimination for /external:env.
//
// RUN: env EXTRA_INCLUDE1="%t/test4/include/y" \
// RUN: env EXTRA_INCLUDE2="%t/test4/include/z;%t/test4/include/y;%t/test4/include/x;%t/test4/include/w" \
// RUN: %clang \
// RUN:     -target x86_64-pc-windows -v -fsyntax-only \
// RUN:     -nostdinc \
// RUN:     -I%t/test4/include/w \
// RUN:     -iexternal %t/test4/include/x \
// RUN:     -iexternal-env=EXTRA_INCLUDE1 \
// RUN:     -iexternal-env=EXTRA_INCLUDE2 \
// RUN:     %t/test4/t.c 2>&1 | FileCheck -DPWD=%t %t/test4/t.c
// RUN: env EXTRA_INCLUDE1="%t/test4/include/y" \
// RUN: env EXTRA_INCLUDE2="%t/test4/include/z;%t/test4/include/y;%t/test4/include/x;%t/test4/include/w" \
// RUN: %clang_cl \
// RUN:     -target x86_64-pc-windows -v -fsyntax-only \
// RUN:     -nobuiltininc /X \
// RUN:     /I%t/test4/include/w \
// RUN:     /external:I %t/test4/include/x \
// RUN:     /external:env:EXTRA_INCLUDE1 \
// RUN:     /external:env:EXTRA_INCLUDE2 \
// RUN:     %t/test4/t.c 2>&1 | FileCheck -DPWD=%t %t/test4/t.c

#--- test4/t.c
#include <a.h>
#include <b.h>
#include <c.h>
#include <d.h>

// CHECK:      ignoring duplicate directory "[[PWD]]/test4/include/y"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/test4/include/x"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/test4/include/w"
// CHECK-NEXT: #include "..." search starts here:
// CHECK-NEXT: #include <...> search starts here:
// CHECK-NEXT: [[PWD]]/test4/include/w
// CHECK-NEXT: [[PWD]]/test4/include/x
// CHECK-NEXT: [[PWD]]/test4/include/y
// CHECK-NEXT: [[PWD]]/test4/include/z
// CHECK-NEXT: End of search list.

#--- test4/include/w/a.h

#--- test4/include/x/a.h
#error 'test4/include/x/a.h' should not have been included!

#--- test4/include/x/b.h

#--- test4/include/y/a.h
#error 'test4/include/y/a.h' should not have been included!

#--- test4/include/y/b.h
#error 'test4/include/y/b.h' should not have been included!

#--- test4/include/y/c.h

#--- test4/include/z/a.h
#error 'test4/include/z/a.h' should not have been included!

#--- test4/include/z/b.h
#error 'test4/include/z/b.h' should not have been included!

#--- test4/include/z/c.h
#error 'test4/include/z/c.h' should not have been included!

#--- test4/include/z/d.h


// Test 5: Validate ordering and duplicate elimination for the INCLUDE and
// EXTERNAL_INCLUDE environment variables.
//
// RUN: env EXTRA_INCLUDE="%t/test5/include/w" \
// RUN: env INCLUDE="%t/test5/include/x;%t/test5/include/y;%t/test5/include/w;%t/test5/include/v;%t/test5/include/u" \
// RUN: env EXTERNAL_INCLUDE="%t/test5/include/z;%t/test5/include/y;%t/test5/include/w;%t/test5/include/v;%t/test5/include/u" \
// RUN: %clang \
// RUN:     -target x86_64-pc-windows -v -fsyntax-only \
// RUN:     -nostdinc \
// RUN:     -I%t/test5/include/u \
// RUN:     -iexternal %t/test5/include/v \
// RUN:     -iexternal-env=EXTRA_INCLUDE \
// RUN:     -iexternal-env=INCLUDE \
// RUN:     -iexternal-env=EXTERNAL_INCLUDE \
// RUN:     %t/test5/t.c 2>&1 | FileCheck -DPWD=%t %t/test5/t.c
// RUN: env EXTRA_INCLUDE="%t/test5/include/w" \
// RUN: env INCLUDE="%t/test5/include/x;%t/test5/include/y;%t/test5/include/w;%t/test5/include/v;%t/test5/include/u" \
// RUN: env EXTERNAL_INCLUDE="%t/test5/include/z;%t/test5/include/y;%t/test5/include/w;%t/test5/include/v;%t/test5/include/u" \
// RUN: %clang_cl \
// RUN:     -target x86_64-pc-windows -v -fsyntax-only \
// RUN:     -nobuiltininc \
// RUN:     /I%t/test5/include/u \
// RUN:     /external:I %t/test5/include/v \
// RUN:     /external:env:EXTRA_INCLUDE \
// RUN:     %t/test5/t.c 2>&1 | FileCheck -DPWD=%t %t/test5/t.c

#--- test5/t.c
#include <a.h>
#include <b.h>
#include <c.h>
#include <d.h>
#include <e.h>
#include <f.h>

// CHECK:      ignoring duplicate directory "[[PWD]]/test5/include/w"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/test5/include/v"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/test5/include/u"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/test5/include/y"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/test5/include/w"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/test5/include/v"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/test5/include/u"
// CHECK-NEXT: #include "..." search starts here:
// CHECK-NEXT: #include <...> search starts here:
// CHECK-NEXT: [[PWD]]/test5/include/u
// CHECK-NEXT: [[PWD]]/test5/include/v
// CHECK-NEXT: [[PWD]]/test5/include/w
// CHECK-NEXT: [[PWD]]/test5/include/x
// CHECK-NEXT: [[PWD]]/test5/include/y
// CHECK-NEXT: [[PWD]]/test5/include/z
// CHECK-NEXT: End of search list.

#--- test5/include/u/a.h

#--- test5/include/v/a.h
#error 'test5/include/v/a.h' should not have been included!

#--- test5/include/v/b.h

#--- test5/include/w/a.h
#error 'test5/include/w/a.h' should not have been included!

#--- test5/include/w/b.h
#error 'test5/include/w/b.h' should not have been included!

#--- test5/include/w/c.h

#--- test5/include/x/a.h
#error 'test5/include/x/a.h' should not have been included!

#--- test5/include/x/b.h
#error 'test5/include/x/b.h' should not have been included!

#--- test5/include/x/c.h
#error 'test5/include/x/c.h' should not have been included!

#--- test5/include/x/d.h

#--- test5/include/y/a.h
#error 'test5/include/y/a.h' should not have been included!

#--- test5/include/y/b.h
#error 'test5/include/y/b.h' should not have been included!

#--- test5/include/y/c.h
#error 'test5/include/y/c.h' should not have been included!

#--- test5/include/y/d.h
#error 'test5/include/y/d.h' should not have been included!

#--- test5/include/y/e.h

#--- test5/include/z/a.h
#error 'test5/include/z/a.h' should not have been included!

#--- test5/include/z/b.h
#error 'test5/include/z/b.h' should not have been included!

#--- test5/include/z/c.h
#error 'test5/include/z/c.h' should not have been included!

#--- test5/include/z/d.h
#error 'test5/include/z/d.h' should not have been included!

#--- test5/include/z/e.h
#error 'test5/include/z/e.h' should not have been included!

#--- test5/include/z/f.h
