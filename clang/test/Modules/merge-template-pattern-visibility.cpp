// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fno-modules-error-recovery -std=c++14 \
// RUN:            -fmodule-name=X -emit-module %S/Inputs/merge-template-pattern-visibility/module.modulemap -x c++ \
// RUN:            -fmodules-local-submodule-visibility -o %t/X.pcm
// RUN: %clang_cc1 -fmodules -fno-modules-error-recovery -std=c++14 \
// RUN:            -fmodule-name=Y -emit-module %S/Inputs/merge-template-pattern-visibility/module.modulemap -x c++ \
// RUN:            -fmodules-local-submodule-visibility -o %t/Y.pcm
// RUN: %clang_cc1 -fmodules -fno-modules-error-recovery -std=c++14 -fmodule-file=%t/X.pcm -fmodule-file=%t/Y.pcm \
// RUN:            -fmodules-local-submodule-visibility -verify %s -I%S/Inputs/merge-template-pattern-visibility
// RUN: %clang_cc1 -fmodules -fno-modules-error-recovery -std=c++14 -fmodule-file=%t/Y.pcm -fmodule-file=%t/X.pcm \
// RUN:            -fmodules-local-submodule-visibility -verify %s -I%S/Inputs/merge-template-pattern-visibility

#include "b.h"
#include "d.h"

// expected-no-diagnostics
void g() {
  CrossModuleMerge::B<int> bi;
  CrossModuleMerge::C(0);
  CrossModuleMerge::D(0);
}
