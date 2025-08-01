// We have to run the compilation step to see the output, so we must be able to
// target Windows.
// REQUIRES: x86-registered-target

// NOTE: -fno-integrated-cc1 has been added to work around an ASAN failure
//       caused by in-process cc1 invocation. Clang InterfaceStubs is not the
//       culprit, but Clang Interface Stubs' Driver pipeline setup uncovers an
//       existing ASAN issue when invoking multiple normal cc1 jobs along with
//       multiple Clang Interface Stubs cc1 jobs together.
//       There is currently a discussion of this going on at:
//         https://reviews.llvm.org/D69825

// RUN: %clang_cl -fno-integrated-cc1 --target=i686-pc-win32 /c /Fo%T/ /showFilenames -- %s 2>&1 | FileCheck -check-prefix=show %s
// RUN: %clang_cl -fno-integrated-cc1 --target=i686-pc-win32 /c /Fo%T/ /showFilenames -- %s %S/Inputs/wildcard*.c 2>&1 | FileCheck -check-prefix=multiple %s

// RUN: %clang_cl -fno-integrated-cc1 --target=i686-pc-win32 /c /Fo%T/ -- %s 2>&1 | FileCheck -check-prefix=noshow %s
// RUN: %clang_cl -fno-integrated-cc1 --target=i686-pc-win32 /c /Fo%T/ /showFilenames /showFilenames- -- %s 2>&1 | FileCheck -check-prefix=noshow %s


#pragma message "Hello"

// show: cl-showfilenames.c
// show-NEXT: warning: Hello

// multiple: cl-showfilenames.c
// multiple-NEXT: warning: Hello
// multiple: wildcard1.c
// multiple-NEXT: wildcard2.c

// noshow: warning: Hello
// noshow-NOT: cl-showfilenames.c
