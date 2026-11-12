// RUN: rm -rf %t
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x c++ -fmodules-cache-path=%t \
// RUN:   -emit-module -fmodule-name=a -o %t/a.pcm \
// RUN:   %S/Inputs/merge-template-friend/module.modulemap
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x c++ -fmodules-cache-path=%t \
// RUN:   -emit-module -fmodule-name=b -o %t/b.pcm \
// RUN:   %S/Inputs/merge-template-friend/module.modulemap
//
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -x c++ -fmodules-cache-path=%t \
// RUN:   -I%S/Inputs/merge-template-friend \
// RUN:   -fmodule-file=%t/a.pcm \
// RUN:   -fmodule-file=%t/b.pcm \
// RUN:   -verify %s

#include "friend.h"
#include "def.h"

::ns::C<int> c;

// expected-no-diagnostics
