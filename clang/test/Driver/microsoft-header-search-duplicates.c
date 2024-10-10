// Test that pruning of header search paths emulates MSVC behavior when in
// Microsoft compatibility mode.
// See header-search-duplicates.c for GCC compatible behavior.

// RUN: rm -rf %t
// RUN: split-file %s %t

// Test the clang driver with a Windows target that implicitly enables the
// -fms-compatibility option. The -nostdinc option is used to suppress default
// search paths to ease testing.
// XUN: env INCLUDE="%t/include/p;%t/include/o" \
// XUN: env EXTRA_INCLUDE="%t/include/t;%t/include/q;%t/include/r" \
// XUN  %clang \
// XUN:     -target x86_64-pc-windows \
// XUN:     -v -fsyntax-only \
// XUN:     -nostdinc \
// XUN:     -isystem %t/include/s \
// XUN:     -isystem %t/include/w \
// XUN:     -I%t/include/v \
// XUN:     -isystem %t/include/x \
// XUN:     -I%t/include/z \
// XUN:     -isystem %t/include/y \
// XUN:     -I%t/include/r \
// XUN:     -external:env:EXTRA_INCLUDE \
// XUN:     -I%t/include/t \
// XUN:     -isystem %t/include/z \
// XUN:     -I%t/include/o \
// XUN:     -I%t/include/x \
// XUN:     -isystem %t/include/y \
// XUN:     -I%t/include/v \
// XUN:     -isystem %t/include/w \
// XUN:     -I%t/include/u \
// XUN:     %t/test.c 2>&1 | FileCheck -DPWD=%t %t/test.c

// Test the clang-cl driver with a Windows target that implicitly enables the
// -fms-compatibility option. The /X option is used instead of -nostdinc
// because the latter option suppresses all system include paths including
// those specified by /external:I. The -nobuiltininc option is uesd to suppress
// the Clang resource directory. The -nostdlibinc option is used to suppress
// search paths for the Windows SDK, CRT, MFC, ATL, etc...
// RUN: env INCLUDE="%t/include/p;%t/include/o" \
// RUN: env EXTRA_INCLUDE="%t/include/t;%t/include/q;%t/include/r" \
// RUN: %clang_cl \
// RUN:     -target x86_64-pc-windows \
// RUN:     -v -fsyntax-only \
// RUN:     -nobuiltininc \
// RUN:     -nostdlibinc \
// RUN:     /external:I %t/include/s \
// RUN:     /external:I %t/include/w \
// RUN:     /I%t/include/v \
// RUN:     /external:I %t/include/x \
// RUN:     /I%t/include/z \
// RUN:     /external:I %t/include/y \
// RUN:     /I%t/include/r \
// RUN:     /external:env:EXTRA_INCLUDE \
// RUN:     /I%t/include/t \
// RUN:     /external:I %t/include/z \
// RUN:     /I%t/include/o \
// RUN:     /I%t/include/x \
// RUN:     /external:I %t/include/y \
// RUN:     /I%t/include/v \
// RUN:     /external:I %t/include/w \
// RUN:     /I%t/include/u \
// RUN:     %t/test.c 2>&1 | FileCheck -DPWD=%t %t/test.c

#--- test.c
#include <a.h>
#include <b.h>
#include <c.h>
#include <d.h>
#include <e.h>
#include <f.h>
#include <g.h>
#include <h.h>
#include <i.h>
#include <j.h>
#include <k.h>
#include <l.h>

// The expected behavior is that user search paths that duplicate a system search
// path are ignored, that user search paths that duplicate a previous user
// search path are ignored, and that system search search paths that duplicate
// a later system search path are ignored.
// CHECK:      ignoring duplicate directory "[[PWD]]/include/v"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/include/x"
// CHECK-NEXT:  as it is a non-system directory that duplicates a system directory
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/include/z"
// CHECK-NEXT:  as it is a non-system directory that duplicates a system directory
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/include/t"
// CHECK-NEXT:  as it is a non-system directory that duplicates a system directory
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/include/r"
// CHECK-NEXT:  as it is a non-system directory that duplicates a system directory
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/include/y"
// CHECK-NEXT: ignoring duplicate directory "[[PWD]]/include/w"
// CHECK:      #include <...> search starts here:
// CHECK-NEXT: [[PWD]]/include/s
// CHECK-NEXT: [[PWD]]/include/v
// CHECK-NEXT: [[PWD]]/include/x
// CHECK-NEXT: [[PWD]]/include/r
// CHECK-NEXT: [[PWD]]/include/t
// CHECK-NEXT: [[PWD]]/include/z
// CHECK-NEXT: [[PWD]]/include/o
// CHECK-NEXT: [[PWD]]/include/y
// CHECK-NEXT: [[PWD]]/include/w
// CHECK-NEXT: [[PWD]]/include/u
// CHECK-NEXT: [[PWD]]/include/q
// CHECK-NEXT: [[PWD]]/include/p
// CHECK-NEXT: End of search list.

#--- include/s/a.h

#--- include/v/a.h
#error 'include/v/a.h' should not have been included!

#--- include/v/b.h

#--- include/x/a.h
#error 'include/x/a.h' should not have been included!

#--- include/x/b.h
#error 'include/x/b.h' should not have been included!

#--- include/x/c.h

#--- include/r/a.h
#error 'include/r/a.h' should not have been included!

#--- include/r/b.h
#error 'include/r/b.h' should not have been included!

#--- include/r/c.h
#error 'include/r/c.h' should not have been included!

#--- include/r/d.h

#--- include/t/a.h
#error 'include/t/a.h' should not have been included!

#--- include/t/b.h
#error 'include/t/b.h' should not have been included!

#--- include/t/c.h
#error 'include/t/c.h' should not have been included!

#--- include/t/d.h
#error 'include/t/cdh' should not have been included!

#--- include/t/e.h

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

#--- include/o/a.h
#error 'include/o/a.h' should not have been included!

#--- include/o/b.h
#error 'include/o/b.h' should not have been included!

#--- include/o/c.h
#error 'include/o/c.h' should not have been included!

#--- include/o/d.h
#error 'include/o/d.h' should not have been included!

#--- include/o/e.h
#error 'include/o/e.h' should not have been included!

#--- include/o/f.h
#error 'include/o/f.h' should not have been included!

#--- include/o/g.h

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
#error 'include/y/f.h' should not have been included!

#--- include/y/g.h
#error 'include/y/g.h' should not have been included!

#--- include/y/h.h

#--- include/w/a.h
#error 'include/w/a.h' should not have been included!

#--- include/w/b.h
#error 'include/w/b.h' should not have been included!

#--- include/w/c.h
#error 'include/w/c.h' should not have been included!

#--- include/w/d.h
#error 'include/w/d.h' should not have been included!

#--- include/w/e.h
#error 'include/w/e.h' should not have been included!

#--- include/w/f.h
#error 'include/w/f.h' should not have been included!

#--- include/w/g.h
#error 'include/w/g.h' should not have been included!

#--- include/w/h.h
#error 'include/w/h.h' should not have been included!

#--- include/w/i.h

#--- include/u/a.h
#error 'include/u/a.h' should not have been included!

#--- include/u/b.h
#error 'include/u/b.h' should not have been included!

#--- include/u/c.h
#error 'include/u/c.h' should not have been included!

#--- include/u/d.h
#error 'include/u/d.h' should not have been included!

#--- include/u/e.h
#error 'include/u/e.h' should not have been included!

#--- include/u/f.h
#error 'include/u/f.h' should not have been included!

#--- include/u/g.h
#error 'include/u/g.h' should not have been included!

#--- include/u/h.h
#error 'include/u/h.h' should not have been included!

#--- include/u/i.h
#error 'include/u/i.h' should not have been included!

#--- include/u/j.h

#--- include/q/a.h
#error 'include/q/a.h' should not have been included!

#--- include/q/b.h
#error 'include/q/b.h' should not have been included!

#--- include/q/c.h
#error 'include/q/c.h' should not have been included!

#--- include/q/d.h
#error 'include/q/d.h' should not have been included!

#--- include/q/e.h
#error 'include/q/e.h' should not have been included!

#--- include/q/f.h
#error 'include/q/f.h' should not have been included!

#--- include/q/g.h
#error 'include/q/g.h' should not have been included!

#--- include/q/h.h
#error 'include/q/h.h' should not have been included!

#--- include/q/i.h
#error 'include/q/i.h' should not have been included!

#--- include/q/j.h
#error 'include/q/j.h' should not have been included!

#--- include/q/k.h

#--- include/p/a.h
#error 'include/p/a.h' should not have been included!

#--- include/p/b.h
#error 'include/p/b.h' should not have been included!

#--- include/p/c.h
#error 'include/p/c.h' should not have been included!

#--- include/p/d.h
#error 'include/p/d.h' should not have been included!

#--- include/p/e.h
#error 'include/p/e.h' should not have been included!

#--- include/p/f.h
#error 'include/p/f.h' should not have been included!

#--- include/p/g.h
#error 'include/p/g.h' should not have been included!

#--- include/p/h.h
#error 'include/p/h.h' should not have been included!

#--- include/p/i.h
#error 'include/p/i.h' should not have been included!

#--- include/p/j.h
#error 'include/p/j.h' should not have been included!

#--- include/p/k.h
#error 'include/p/k.h' should not have been included!

#--- include/p/l.h
