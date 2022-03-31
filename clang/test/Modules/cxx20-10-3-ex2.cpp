// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-3-ex2-tu1.cpp \
// RUN:  -o %t/M.pcm

// RUN: %clang_cc1 -std=c++20 -S %t/std10-3-ex2-tu2.cpp \
// RUN:  -fmodule-file=%t/M.pcm -o %t/tu_8.s -verify

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-3-ex2-tu3.cpp \
// RUN:  -o %t/M.pcm -verify

//--- std10-3-ex2-tu1.cpp
export module M;

//--- std10-3-ex2-tu2.cpp
module M;
          // error: cannot import M in its own unit
import M; // expected-error {{import of module 'M' appears within its own implementation}}

//--- std10-3-ex2-tu3.cpp
export module M;
          // error: cannot import M in its own unit
import M; // expected-error {{import of module 'M' appears within its own interface}}
