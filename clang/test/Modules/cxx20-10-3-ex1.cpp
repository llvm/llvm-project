// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-3-ex1-tu1.cpp \
// RUN:  -o %t/M_PartImpl.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-3-ex1-tu2.cpp \
// RUN:  -fmodule-file=%t/M_PartImpl.pcm -o %t/M.pcm -verify

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-3-ex1-tu3.cpp \
// RUN:  -o %t/M_Part.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/std10-3-ex1-tu4.cpp \
// RUN:  -fmodule-file=%t/M_Part.pcm -o %t/M.pcm

//--- std10-3-ex1-tu1.cpp
module M:PartImpl;

// expected-no-diagnostics

//--- std10-3-ex1-tu2.cpp
export module M;
                     // error: exported partition :Part is an implementation unit
export import :PartImpl; // expected-error {{module partition implementations cannot be exported}}

//--- std10-3-ex1-tu3.cpp
export module M:Part;

// expected-no-diagnostics

//--- std10-3-ex1-tu4.cpp
export module M;
export import :Part;

// expected-no-diagnostics
