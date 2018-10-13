// We have to run the compilation step to see the output, so we must be able to
// target Windows.
// REQUIRES: x86-registered-target

// RUN: %clang_cl --target=i686-pc-win32 /c /Fo%T/ /showFilenames -- %s 2>&1 | FileCheck -check-prefix=show %s
// RUN: %clang_cl --target=i686-pc-win32 /c /Fo%T/ /showFilenames -- %s %S/Inputs/wildcard*.c 2>&1 | FileCheck -check-prefix=multiple %s

// RUN: %clang_cl --target=i686-pc-win32 /c /Fo%T/ -- %s 2>&1 | FileCheck -check-prefix=noshow %s
// RUN: %clang_cl --target=i686-pc-win32 /c /Fo%T/ /showFilenames /showFilenames- -- %s 2>&1 | FileCheck -check-prefix=noshow %s


#pragma message "Hello"

// show: cl-showfilenames.c
// show-NEXT: warning: Hello

// multiple: cl-showfilenames.c
// multiple-NEXT: warning: Hello
// multiple: wildcard1.c
// multiple-NEXT: wildcard2.c

// noshow: warning: Hello
// noshow-NOT: cl-showfilenames.c
