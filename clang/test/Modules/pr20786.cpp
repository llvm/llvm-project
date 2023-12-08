// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fmodule-name=TBranchProxy -emit-module -x c++ %S/Inputs/PR20786/module.modulemap
