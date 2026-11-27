// This test checks that explicitly building the same module from different
// working directories results in the same PCM contents.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: mkdir %t/one
// RUN: mkdir %t/two

//--- module.modulemap
module M { header "M.h" }

//--- M.h

// RUN: cd %t/one && %clang_cc1 -fmodules -emit-module %t/module.modulemap -fmodule-name=M -o %t/M_one.pcm
// RUN: cd %t/two && %clang_cc1 -fmodules -emit-module %t/module.modulemap -fmodule-name=M -o %t/M_two.pcm

// RUN: diff %t/M_one.pcm %t/M_two.pcm
