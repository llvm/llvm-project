// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fno-modules-error-recovery -std=c++14 \
// RUN:            -fmodule-name=X -emit-module %S/Inputs/merge-template-pattern-visibility/module.modulemap -x c++ \
// RUN:            -fmodules-local-submodule-visibility -o %t/X.pcm
