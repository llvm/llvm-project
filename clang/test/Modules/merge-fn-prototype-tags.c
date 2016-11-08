// RUN: rm -rf %t
// RUN: %clang_cc1 -x c -I%S/Inputs/merge-fn-prototype-tags -verify %s
// RUN: %clang_cc1 -fmodules -fmodule-map-file=%S/Inputs/merge-fn-prototype-tags/module.modulemap -fmodules-cache-path=%t -x c -I%S/Inputs/merge-fn-prototype-tags -verify %s

#include "c.h"
void mmalloc_attach() { struct stat sbuf; }

// expected-no-diagnostics
