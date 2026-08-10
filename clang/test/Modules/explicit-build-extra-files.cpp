// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cd %t
// RUN: echo 'module X {}' > %t/x
// RUN: echo 'module Y {}' > %t/y
//
// RUN: %clang_cc1 -emit-module -fmodules -fmodule-name=X %t/x -x c++ -o %t/x.pcm
// RUN: %clang_cc1 -emit-module -fmodules -fmodule-name=Y %t/y -x c++ -o %t/y.pcm
// RUN: %clang_cc1 -fmodules -fmodule-file=%t/x.pcm -fmodule-file=%t/y.pcm -x c++ /dev/null -fsyntax-only
//
// RUN: not test -f %t/modules.timestamp
// RUN: not test -f %t/modules.idx
