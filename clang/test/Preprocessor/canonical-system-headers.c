// don't create symlinks on windows
// UNSUPPORTED: system-windows
// REQUIRES: shell

// RUN: rm -rf %t
// RUN: mkdir -p %t/foo/
// RUN: ln -f -s %S/Inputs/canonical-system-headers %t/foo/include
// RUN: %clang_cc1 -isystem %t/foo/include -sys-header-deps -MT foo.o -dependency-file %t2 %s -fsyntax-only
// RUN: FileCheck %s --check-prefix=NOCANON --implicit-check-not=a.h < %t2
// RUN: %clang_cc1 -isystem %t/foo/include -sys-header-deps -MT foo.o -dependency-file %t2 %s -fsyntax-only -canonical-system-headers
// RUN: FileCheck %s --check-prefix=CANON --implicit-check-not=a.h < %t2

// NOCANON: foo/include/a.h
// CANON: Inputs/canonical-system-headers/a.h

#include <a.h>
