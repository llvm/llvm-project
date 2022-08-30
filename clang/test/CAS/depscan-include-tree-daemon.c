// REQUIRES: system-darwin, clang-cc1daemon
// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang -cc1depscan -o %t/inline.rsp -fdepscan=inline -fdepscan-include-tree -cc1-args -cc1 -triple x86_64-apple-macos11.0 \
// RUN:     -fsyntax-only %t/t.c -I %t/includes -isysroot %S/Inputs/SDK -fcas-path %t/cas -DSOME_MACRO -dependency-file %t/inline.d -MT deps

// RUN: %clang -cc1depscand -start %{clang-daemon-dir}/depscan-include-tree-daemon -cas-args -fcas-path %t/cas -faction-cache-path %t/cache
// RUN: %clang -cc1depscan -o %t/daemon.rsp -fdepscan=daemon -fdepscan-include-tree -cc1-args -cc1 -triple x86_64-apple-macos11.0 \
// RUN:     -fsyntax-only %t/t.c -I %t/includes -isysroot %S/Inputs/SDK -fcas-path %t/cas -DSOME_MACRO -dependency-file %t/daemon.d -MT deps
// RUN: %clang -cc1depscand -shutdown %{clang-daemon-dir}/depscan-include-tree-daemon

// RUN: diff -u %t/inline.rsp %t/daemon.rsp
// RUN: diff -u %t/inline.d %t/daemon.d

//--- t.c
#include "t.h"

int test(struct S *s) {
  return s->x;
}

//--- includes/t.h
struct S {
  int x;
};
