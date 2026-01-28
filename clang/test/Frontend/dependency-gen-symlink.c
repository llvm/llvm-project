// REQUIRES: symlinks

// Basic test
// RUN: rm -rf %t.dir
// RUN: mkdir %t.dir
// RUN: mkdir %t.dir/a
// RUN: mkdir %t.dir/b
// RUN: echo "#ifndef HEADER_A" > %t.dir/a/header.h
// RUN: echo "#define HEADER_A" >> %t.dir/a/header.h
// RUN: echo "#endif" >> %t.dir/a/header.h
// RUN: ln -s %t.dir/a/header.h %t.dir/b/header.h

// RUN: %clang_cc1 -dependency-file %t.dir/file.deps -MT %s.o %s -fsyntax-only -I %t.dir -isystem %S/Inputs/SystemHeaderPrefix
// RUN: FileCheck -input-file=%t.dir/file.deps %s
// CHECK: dependency-gen-symlink.c.o
// CHECK: dependency-gen-symlink.c
// CHECK: a/header.h
// CHECK: b/header.h
// CHECK-NOT: with-header-guard.h
#include "a/header.h"
#include "b/header.h"
// System header shouldn't be included in dependencies.
#include <with-header-guard.h>
#include <with-header-guard.h>
