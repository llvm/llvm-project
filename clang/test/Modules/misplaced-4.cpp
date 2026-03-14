// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -emit-module -fmodule-name=Misplaced -fmodules-cache-path=%t -x c++ -I %S/Inputs %S/Inputs/misplaced/misplaced.modulemap -verify
