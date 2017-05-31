// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fno-modules-error-recovery \
// RUN:            -fmodule-name=X -emit-module %S/Inputs/merge-template-pattern-visibility/module.modulemap -x c++ \
// RUN:            -fmodules-local-submodule-visibility
