// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-strict-decluse -I%S/Inputs/preprocess-decluse \
// RUN:            -fmodule-name=B -emit-module -o %t/b.pcm \
// RUN:            -fmodule-map-file=%S/Inputs/preprocess-decluse/a.modulemap \
// RUN:            -x c++-module-map %S/Inputs/preprocess-decluse/b.modulemap
// RUN: %clang_cc1 -fmodules -fmodules-strict-decluse -I%S/Inputs/preprocess-decluse \
// RUN:            -fmodule-map-file=%S/Inputs/preprocess-decluse/main.modulemap \
// RUN:            -fmodule-file=%t/b.pcm -fmodule-name=Main %s -verify
// RUN: %clang_cc1 -fmodules -fmodules-strict-decluse -I%S/Inputs/preprocess-decluse \
// RUN:            -fmodule-map-file=%S/Inputs/preprocess-decluse/main.modulemap \
// RUN:            -fmodule-file=%t/b.pcm -fmodule-name=Main %s \
// RUN:            -E -frewrite-imports -o %t/rewrite.ii
// RUN: %clang_cc1 -fmodules -fmodules-strict-decluse -I%S/Inputs/preprocess-decluse \
// RUN:            -fmodule-map-file=%S/Inputs/preprocess-decluse/main.modulemap \
// RUN:            -fmodule-name=Main %t/rewrite.ii -verify

// expected-no-diagnostics
#include "b.h"
