// REQUIRES: x86
// RUN: split-file %s %t.dir

// RUN: llvm-mc -filetype=obj -triple=x86_64-win32-gnu %t.dir/main.s -o %t.main.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-win32-gnu %t.dir/lib1.s -o %t.lib1.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-win32-gnu %t.dir/lib2.s -o %t.lib2.o

// RUN: rm -f %t.lib.a
// RUN: llvm-ar cru %t.lib.a %t.lib1.o %t.lib2.o
// RUN: lld-link -dll -out:%t-1.dll -entry:entry %t.main.o %t.lib.a
// RUN: lld-link -dll -out:%t-2.dll -entry:entry %t.main.o %t.lib.a -includeoptional:libfunc

// RUN: llvm-readobj --coff-exports %t-1.dll | FileCheck --implicit-check-not=Name: %s --check-prefix=CHECK-DEFAULT
// RUN: llvm-readobj --coff-exports %t-2.dll | FileCheck --implicit-check-not=Name: %s --check-prefix=CHECK-INCLUDEOPTIONAL

// CHECK-DEFAULT: Name:
// CHECK-DEFAULT: Name: myfunc

// CHECK-INCLUDEOPTIONAL: Name:
// CHECK-INCLUDEOPTIONAL: Name: libfunc
// CHECK-INCLUDEOPTIONAL: Name: myfunc
// CHECK-INCLUDEOPTIONAL: Name: otherlibfunc

#--- main.s
.global entry
entry:
  ret

.global myfunc
myfunc:
  ret

.section .drectve
.ascii "-export:myfunc "

#--- lib1.s
.global libfunc
libfunc:
  call otherlibfunc
  ret

.section .drectve
.ascii "-export:libfunc "

#--- lib2.s
.global otherlibfunc
otherlibfunc:
  ret

.section .drectve
.ascii "-export:otherlibfunc "
