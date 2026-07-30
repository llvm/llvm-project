// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo 'module tmp { header "tmp.h" }' > %t/map
// RUN: touch %t/tmp.h
// RUN: %clang_cc1 -fmodules -DFOO=1 -x c++ -fmodule-name=tmp %t/map -emit-module -o %t/tmp.pcm

// Can use the module.
// RUN: %clang_cc1 -fmodules -DFOO=1 -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp.pcm -verify -I%t %s

// Can use the module if an input file is newer. (This happens on remote file systems.)
// RUN: sleep 1
// RUN: touch %t/tmp.h
// RUN: %clang_cc1 -fmodules -DFOO=1 -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp.pcm -verify -I%t %s

// Can use the module if -D flags change.
// RUN: %clang_cc1 -fmodules -DFOO=2 -DBAR=1 -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp.pcm -verify -I%t %s
// RUN: %clang_cc1 -fmodules -DBAR=2 -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp.pcm -verify -I%t %s

// Can use the module if -W flags change.
// RUN: %clang_cc1 -fmodules -DBAR=2 -Wextra -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp.pcm -verify -I%t %s

// Can use the module if -I flags change.
// RUN: %clang_cc1 -fmodules -DBAR=2 -I. -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp.pcm -verify -I%t %s

// Can use the module if -fPIC/-fPIE flags change.
// RUN: %clang_cc1 -fmodules -DBAR=2 -pic-level 2 -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp.pcm -verify -I%t %s
// RUN: %clang_cc1 -fmodules -DBAR=2 -pic-level 1 -pic-is-pie -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp.pcm -verify -I%t %s

// Can use the module if -static flag changes.
// RUN: %clang_cc1 -fmodules -DBAR=2 -static-define -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp.pcm -verify -I%t %s

// Can use the module if -fsanitize= flags change.
// RUN: %clang_cc1 -fmodules -DBAR=2 -fsanitize=address -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp.pcm -verify -I%t %s
//
// RUN: %clang_cc1 -fmodules -DFOO=1 -fsanitize=address -x c++ -fmodule-name=tmp %t/map -emit-module -o %t/tmp-san.pcm
// RUN: %clang_cc1 -fmodules -DBAR=2 -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp-san.pcm -verify -I%t %s

// -fno-assume-sane-operator-new is implied by the driver -fsanitize=address flag.
// RUN: %clang_cc1 -fmodules -DBAR=2 -fno-assume-sane-operator-new -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp.pcm -verify -I%t %s

// Can use the module if -O flags change.
// RUN: %clang_cc1 -fmodules -DBAR=2 -Os -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp.pcm -verify -I%t %s
//
// RUN: %clang_cc1 -fmodules -DFOO=1 -O2 -x c++ -fmodule-name=tmp %t/map -emit-module -o %t/tmp-O2.pcm
// RUN: %clang_cc1 -fmodules -DBAR=2 -O0 -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp-O2.pcm -verify -I%t %s
// RUN: %clang_cc1 -fmodules -DBAR=2 -Os -x c++ -fmodule-map-file=%t/map -fmodule-file=%t/tmp-O2.pcm -verify -I%t %s

#include "tmp.h" // expected-no-diagnostics

#ifndef BAR
#if FOO != 1
#error bad FOO from command line and module
#endif
#elif BAR == 1
#if FOO != 2
#error bad FOO from command line overriding module
#endif
#elif BAR == 2
#ifdef FOO
#error FOO leaked from module
#endif
#else
#error bad BAR
#endif
