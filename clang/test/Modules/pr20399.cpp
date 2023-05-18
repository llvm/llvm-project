// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -fmodule-name=libGdml -emit-module -x c++ -std=c++11 %S/Inputs/PR20399/module.modulemap
