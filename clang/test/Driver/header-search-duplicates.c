// Test that pruning of header search paths emulates GCC behavior when not in
// Microsoft compatibility mode.
// See microsoft-header-search-duplicates.c for Microsoft compatible behavior.

// RUN: rm -rf %t
// RUN: split-file %s %t

// This test uses the -nostdinc option to suppress default search paths to
// ease testing.

// Header search paths are categorized into the following general groups.
// - Quoted: Search paths that are only used to resolve inclusion of header
//   files specified with quoted inclusion ('#include "X"'). Paths nominated
//   by the '-iquoted' option are added to this group.
// - Angled: Search paths used to resolve inclusion of header files specified
//   with angled inclusion ('#include <X>') or quoted inclusion if a match
//   was not found in the Quoted group. Paths nominated by the '-I',
//   '-iexternal', '-iexternal-env=', and '-iwithprefixbefore' options are
//   added to this group.
// - System: Search paths used to resolve inclusion of a header file for which
//   a match is not found in the Quoted or Angled groups. Paths nominated by
//   the '-dirafter', '-isystem', '-isystem-after', '-iwithprefix', and
//   related language specific options are added to this group.
// Duplicate search paths are identified and processed as follows:
// 1) Paths in the Quoted group that duplicate a previous path in the Quoted
//    group are removed.
// 2) Paths in the Angled group that are duplicated by an external path
//    (as nominated by the '-iexternal' or '-iexternal-env=' options) in the
//    Angled group (regardless of the relative order of the paths) or by a
//    path in the System group are removed
// 3) Paths in the Angled or System groups that duplicate a previous path in
//    the Angled or System group are removed.


// Test 1: Validate ordering and duplicate elimination in the Quoted group.
// This test exhibits a behavioral difference between GCC and Clang. GCC
// removes the last path in the quoted group if it matches the first path
// in the angled group. Clang does not. The difference is observable via
// '#include_next' as this test demonstrates. Clang's behavior makes use of
// '#include_next' across the Quoted and Angled groups reliable regardless
// of whether there is an intervening search path present at the start of
// the Angled group.
//
// RUN: %clang \
// RUN:     -target x86_64-unknown-linux-gnu -v -fsyntax-only \
// RUN:     -nostdinc \
// RUN:     -iquote %t/test1/include/x \
// RUN:     -iquote %t/test1/include/y \
// RUN:     -iquote %t/test1/include/x \
// RUN:     -iquote %t/test1/include/z \
// RUN:     -I%t/test1/include/z \
// RUN:     -I%t/test1/include/y \
// RUN:     %t/test1/t.c 2>&1 | FileCheck -DPWD=%t %t/test1/t.c

#--- test1/t.c
#include "a.h"
#include "b.h"
#include "c.h"

// CHECK:      ignoring duplicate directory "[[PWD]]/test1/include/x"
// CHECK-NEXT: #include "..." search starts here:
// CHECK-NEXT: [[PWD]]/test1/include/x
// CHECK-NEXT: [[PWD]]/test1/include/y
// CHECK-NEXT: [[PWD]]/test1/include/z
// CHECK-NEXT: #include <...> search starts here:
// CHECK-NEXT: [[PWD]]/test1/include/z
// CHECK-NEXT: [[PWD]]/test1/include/y
// CHECK-NEXT: End of search list.

#--- test1/include/x/a.h

#--- test1/include/y/a.h
#error 'test1/include/y/a.h' should not have been included!

#--- test1/include/y/b.h
#if !defined(Y_B_DEFINED)
#define Y_B_DEFINED
#include_next <b.h>
#endif

#--- test1/include/z/a.h
#error 'test1/include/z/a.h' should not have been included!

#--- test1/include/z/b.h
#if !defined(Y_B_DEFINED)
#error 'Y_B_DEFINED' is not defined in test1/include/z/b.h!
#endif

#--- test1/include/z/c.h
#if !defined(Z_C_DEFINED)
#define Z_C_DEFINED
#include_next <c.h>
#endif


// Test 2: Validate ordering and duplicate elimination in the Angled group.
//
// RUN: %clang \
// RUN:     -target x86_64-unknown-linux-gnu -v -fsyntax-only \
// RUN:     -nostdinc \
// RUN:     -iprefix %t/ \
// RUN:     -I%t/test2/include/v \
// RUN:     -iwithprefixbefore test2/include/y \
// RUN:     -I%t/test2/include/u \
// RUN:     -iexternal %t/test2/include/v \
// RUN:     -iwithprefixbefore test2/include/z \
// RUN:     -iexternal %t/test2/include/w \
// RUN:     -I%t/test2/include/x \
// RUN:     -iexternal %t/test2/include/y \
// RUN:     -iwithprefixbefore test2/include/x \
// RUN:     %t/test2/t.c 2>&1 | FileCheck -DPWD=%t %t/test2/t.c

#--- test2/t.c
#include <a.h>
#include <b.h>
#include <c.h>
#include <d.h>
#include <e.h>
#include <f.h>

// CHECK:      ignoring duplicate directory "[[PWD]]/test2/include/v"
// CHECK-NEXT:  as it is a non-system directory that duplicates a system directory
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/test2/include/y"
// CHECK-NEXT:  as it is a non-system directory that duplicates a system directory
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/test2/include/x"
// CHECK-NEXT: #include "..." search starts here:
// CHECK-NEXT: #include <...> search starts here:
// CHECK-NEXT: [[PWD]]/test2/include/u
// CHECK-NEXT: [[PWD]]/test2/include/v
// CHECK-NEXT: [[PWD]]/test2/include/w
// CHECK-NEXT: [[PWD]]/test2/include/x
// CHECK-NEXT: [[PWD]]/test2/include/y
// CHECK-NEXT: [[PWD]]/test2/include/z
// CHECK-NEXT: End of search list.

#--- test2/include/u/a.h

#--- test2/include/v/a.h
#error 'test2/include/v/a.h' should not have been included!

#--- test2/include/v/b.h

#--- test2/include/w/a.h
#error 'test2/include/w/a.h' should not have been included!

#--- test2/include/w/b.h
#error 'test2/include/w/b.h' should not have been included!

#--- test2/include/w/c.h

#--- test2/include/x/a.h
#error 'test2/include/x/a.h' should not have been included!

#--- test2/include/x/b.h
#error 'test2/include/x/b.h' should not have been included!

#--- test2/include/x/c.h
#error 'test2/include/x/c.h' should not have been included!

#--- test2/include/x/d.h

#--- test2/include/y/a.h
#error 'test2/include/y/a.h' should not have been included!

#--- test2/include/y/b.h
#error 'test2/include/y/b.h' should not have been included!

#--- test2/include/y/c.h
#error 'test2/include/y/c.h' should not have been included!

#--- test2/include/y/d.h
#error 'test2/include/y/d.h' should not have been included!

#--- test2/include/y/e.h

#--- test2/include/z/a.h
#error 'test2/include/z/a.h' should not have been included!

#--- test2/include/z/b.h
#error 'test2/include/z/b.h' should not have been included!

#--- test2/include/z/c.h
#error 'test2/include/z/c.h' should not have been included!

#--- test2/include/z/d.h
#error 'test2/include/z/d.h' should not have been included!

#--- test2/include/z/e.h
#error 'test2/include/z/e.h' should not have been included!

#--- test2/include/y/f.h


// Test 3: Validate ordering and duplicate elimination across the Angled and
// System groups.
//
// RUN: %clang \
// RUN:     -target x86_64-unknown-linux-gnu -v -fsyntax-only \
// RUN:     -nostdinc \
// RUN:     -I%t/test3/include/y \
// RUN:     -iexternal %t/test3/include/u \
// RUN:     -I%t/test3/include/v \
// RUN:     -isystem %t/test3/include/y \
// RUN:     -iexternal %t/test3/include/w \
// RUN:     -isystem %t/test3/include/z \
// RUN:     -I%t/test3/include/x \
// RUN:     -isystem %t/test3/include/u \
// RUN:     -iexternal %t/test3/include/x \
// RUN:     %t/test3/t.c 2>&1 | FileCheck -DPWD=%t %t/test3/t.c

#--- test3/t.c
#include <a.h>
#include <b.h>
#include <c.h>
#include <d.h>
#include <e.h>
#include <f.h>

// CHECK:      ignoring duplicate directory "[[PWD]]/test3/include/x"
// CHECK-NEXT:  as it is a non-system directory that duplicates a system directory
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/test3/include/y"
// CHECK-NEXT:  as it is a non-system directory that duplicates a system directory
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/test3/include/u"
// CHECK-NEXT: #include "..." search starts here:
// CHECK-NEXT: #include <...> search starts here:
// CHECK-NEXT: [[PWD]]/test3/include/u
// CHECK-NEXT: [[PWD]]/test3/include/v
// CHECK-NEXT: [[PWD]]/test3/include/w
// CHECK-NEXT: [[PWD]]/test3/include/x
// CHECK-NEXT: [[PWD]]/test3/include/y
// CHECK-NEXT: [[PWD]]/test3/include/z
// CHECK-NEXT: End of search list.

#--- test3/include/u/a.h

#--- test3/include/v/a.h
#error 'test3/include/v/a.h' should not have been included!

#--- test3/include/v/b.h

#--- test3/include/w/a.h
#error 'test3/include/w/a.h' should not have been included!

#--- test3/include/w/b.h
#error 'test3/include/w/b.h' should not have been included!

#--- test3/include/w/c.h

#--- test3/include/x/a.h
#error 'test3/include/x/a.h' should not have been included!

#--- test3/include/x/b.h
#error 'test3/include/x/b.h' should not have been included!

#--- test3/include/x/c.h
#error 'test3/include/x/c.h' should not have been included!

#--- test3/include/x/d.h

#--- test3/include/y/a.h
#error 'test3/include/y/a.h' should not have been included!

#--- test3/include/y/b.h
#error 'test3/include/y/b.h' should not have been included!

#--- test3/include/y/c.h
#error 'test3/include/y/c.h' should not have been included!

#--- test3/include/y/d.h
#error 'test3/include/y/d.h' should not have been included!

#--- test3/include/y/e.h

#--- test3/include/z/a.h
#error 'test3/include/z/a.h' should not have been included!

#--- test3/include/z/b.h
#error 'test3/include/z/b.h' should not have been included!

#--- test3/include/z/c.h
#error 'test3/include/z/c.h' should not have been included!

#--- test3/include/z/d.h
#error 'test3/include/z/d.h' should not have been included!

#--- test3/include/z/e.h
#error 'test3/include/z/e.h' should not have been included!

#--- test3/include/z/f.h


// Test 4: Validate ordering and duplicate elimination across the Angled and
// System groups.
//
// RUN: env EXTRA_INCLUDE="%t/test4/include/w" \
// RUN: %clang \
// RUN:     -target x86_64-unknown-linux-gnu -v -fsyntax-only \
// RUN:     -nostdinc \
// RUN:     -I%t/test4/include/z \
// RUN:     -iexternal %t/test4/include/v \
// RUN:     -iexternal-env=EXTRA_INCLUDE \
// RUN:     -isystem %t/test4/include/x \
// RUN:     -isystem %t/test4/include/y \
// RUN:     -isystem %t/test4/include/x \
// RUN:     -isystem %t/test4/include/w \
// RUN:     -isystem %t/test4/include/v \
// RUN:     -isystem %t/test4/include/z \
// RUN:     %t/test4/t.c 2>&1 | FileCheck -DPWD=%t %t/test4/t.c

#--- test4/t.c
#include <a.h>
#include <b.h>
#include <c.h>
#include <d.h>
#include <e.h>

// CHECK:      ignoring duplicate directory "[[PWD]]/test4/include/x"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/test4/include/w"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/test4/include/v"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/test4/include/z"
// CHECK-NEXT:  as it is a non-system directory that duplicates a system directory
// CHECK-NEXT: #include "..." search starts here:
// CHECK-NEXT: #include <...> search starts here:
// CHECK-NEXT: [[PWD]]/test4/include/v
// CHECK-NEXT: [[PWD]]/test4/include/w
// CHECK-NEXT: [[PWD]]/test4/include/x
// CHECK-NEXT: [[PWD]]/test4/include/y
// CHECK-NEXT: [[PWD]]/test4/include/z
// CHECK-NEXT: End of search list.

#--- test4/include/v/a.h

#--- test4/include/w/a.h
#error 'test4/include/w/a.h' should not have been included!

#--- test4/include/w/b.h

#--- test4/include/x/a.h
#error 'test4/include/x/a.h' should not have been included!

#--- test4/include/x/b.h
#error 'test4/include/x/b.h' should not have been included!

#--- test4/include/x/c.h

#--- test4/include/y/a.h
#error 'test4/include/y/a.h' should not have been included!

#--- test4/include/y/b.h
#error 'test4/include/y/b.h' should not have been included!

#--- test4/include/y/c.h
#error 'test4/include/y/c.h' should not have been included!

#--- test4/include/y/d.h

#--- test4/include/z/a.h
#error 'test4/include/z/a.h' should not have been included!

#--- test4/include/z/b.h
#error 'test4/include/z/b.h' should not have been included!

#--- test4/include/z/c.h
#error 'test4/include/z/c.h' should not have been included!

#--- test4/include/z/d.h
#error 'test4/include/z/d.h' should not have been included!

#--- test4/include/z/e.h
