// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface  %t/a.cpp -o %t/a.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface  %t/c.cpp \
// RUN: -fmodule-file=a=%t/a.pcm -o %t/c.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface  %t/d.cpp \
// RUN: -fmodule-file=a=%t/a.pcm -o %t/d.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface  %t/e.cpp \
// RUN: -fmodule-file=a=%t/a.pcm -o %t/e.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface  %t/a-part.cpp \
// RUN: -o %t/a-part.pcm

// RUN: %clang_cc1 -std=c++20 -emit-module-interface  %t/f.cpp \
// RUN: -fmodule-file=a=%t/a.pcm -o %t/f.pcm -verify

// RUN: %clang_cc1 -std=c++20 -emit-module-interface  %t/g.cpp \
// RUN: -fmodule-file=a=%t/a.pcm -o %t/g.pcm -verify

// Test again with reduced BMI.
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface  %t/a.cpp -o %t/a.pcm

// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface  %t/c.cpp \
// RUN: -fmodule-file=a=%t/a.pcm -o %t/c.pcm

// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface  %t/d.cpp \
// RUN: -fmodule-file=a=%t/a.pcm -o %t/d.pcm

// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface  %t/e.cpp \
// RUN: -fmodule-file=a=%t/a.pcm -o %t/e.pcm

// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface  %t/a-part.cpp \
// RUN: -o %t/a-part.pcm

// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface  %t/f.cpp \
// RUN: -fmodule-file=a=%t/a.pcm -o %t/f.pcm -verify

// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface  %t/g.cpp \
// RUN: -fmodule-file=a=%t/a.pcm -o %t/g.pcm -verify

//--- a.cpp
export module a;

//--- b.hpp
import a;

//--- c.cpp
module;
#include "b.hpp"
export module c;

//--- d.cpp
module;
import a;

export module d;

//--- e.cpp
export module e;

module :private;
import a;

//--- a-part.cpp
export module a:part;

//--- f.cpp
module;
import :part ; // expected-error {{module partition imports cannot be in the global module fragment}}

export module f;

//--- g.cpp

export module g;
module :private;
import :part; // expected-error {{module partition imports cannot be in the private module fragment}}
