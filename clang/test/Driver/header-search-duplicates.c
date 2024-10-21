// Test that pruning of header search paths emulates GCC behavior when not in
// Microsoft compatibility mode.
// See microsoft-header-search-duplicates.c for Microsoft compatible behavior.

// RUN: rm -rf %t
// RUN: split-file %s %t

// Test the clang driver with a target that does not implicitly enable the
// -fms-compatibility option. The -nostdinc option is used to suppress default
// search paths to ease testing.
// RUN: %clang \
// RUN:     -target x86_64-unknown-linux-gnu \
// RUN:     -v -fsyntax-only \
// RUN:     -nostdinc \
// RUN:     -isystem %t/include/S \
// RUN:     -isystem %t/include/w \
// RUN:     -I%t/include/v \
// RUN:     -isystem %t/include/x \
// RUN:     -I%t/include/z \
// RUN:     -isystem %t/include/y \
// RUN:     -isystem %t/include/z \
// RUN:     -I%t/include/x \
// RUN:     -isystem %t/include/y \
// RUN:     -I%t/include/v \
// RUN:     -isystem %t/include/w \
// RUN:     -I%t/include/U \
// RUN:     %t/test.c 2>&1 | FileCheck -DPWD=%t %t/test.c

#--- test.c
#include <a.h>
#include <b.h>
#include <c.h>
#include <d.h>
#include <e.h>
#include <f.h>
#include <g.h>

// The expected behavior is that user search paths are ordered before system
// search paths, that user search paths that duplicate a (later) system search
// path are ignored, and that search paths that duplicate an earlier search
// path of the same user/system kind are ignored.
// CHECK:      ignoring duplicate directory "[[PWD]]/include/v"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/include/x"
// CHECK-NEXT:  as it is a non-system directory that duplicates a system directory
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/include/z"
// CHECK-NEXT:  as it is a non-system directory that duplicates a system directory
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/include/y"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/include/w"
// CHECK:      #include <...> search starts here:
// CHECK-NEXT: [[PWD]]/include/v
// CHECK-NEXT: [[PWD]]/include/U
// CHECK-NEXT: [[PWD]]/include/S
// CHECK-NEXT: [[PWD]]/include/w
// CHECK-NEXT: [[PWD]]/include/x
// CHECK-NEXT: [[PWD]]/include/y
// CHECK-NEXT: [[PWD]]/include/z
// CHECK-NEXT: End of search list.

#--- include/v/a.h

#--- include/U/a.h
#error 'include/U/a.h' should not have been included!

#--- include/U/b.h

#--- include/S/a.h
#error 'include/U/a.h' should not have been included!

#--- include/S/b.h
#error 'include/S/b.h' should not have been included!

#--- include/S/c.h

#--- include/w/a.h
#error 'include/w/a.h' should not have been included!

#--- include/w/b.h
#error 'include/w/b.h' should not have been included!

#--- include/w/c.h
#error 'include/w/c.h' should not have been included!

#--- include/w/d.h

#--- include/x/a.h
#error 'include/x/a.h' should not have been included!

#--- include/x/b.h
#error 'include/x/b.h' should not have been included!

#--- include/x/c.h
#error 'include/x/c.h' should not have been included!

#--- include/x/d.h
#error 'include/x/d.h' should not have been included!

#--- include/x/e.h

#--- include/y/a.h
#error 'include/y/a.h' should not have been included!

#--- include/y/b.h
#error 'include/y/b.h' should not have been included!

#--- include/y/c.h
#error 'include/y/c.h' should not have been included!

#--- include/y/d.h
#error 'include/y/d.h' should not have been included!

#--- include/y/e.h
#error 'include/y/e.h' should not have been included!

#--- include/y/f.h

#--- include/z/a.h
#error 'include/z/a.h' should not have been included!

#--- include/z/b.h
#error 'include/z/b.h' should not have been included!

#--- include/z/c.h
#error 'include/z/c.h' should not have been included!

#--- include/z/d.h
#error 'include/z/d.h' should not have been included!

#--- include/z/e.h
#error 'include/z/e.h' should not have been included!

#--- include/z/f.h
#error 'include/z/f.h' should not have been included!

#--- include/z/g.h
