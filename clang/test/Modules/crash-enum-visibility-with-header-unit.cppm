// Fixes #165445

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -x c++-user-header %t/header.h \
// RUN:   -emit-header-unit -o %t/header.pcm
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -fmodule-file=%t/header.pcm \
// RUN:   -emit-module-interface -o %t/A.pcm
// 
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -fmodule-file=%t/header.pcm \
// RUN:   -emit-module-interface -o %t/B.pcm
//
// RUN: %clang_cc1 -std=c++20 %t/use.cpp \
// RUN:   -fmodule-file=A=%t/A.pcm -fmodule-file=B=%t/B.pcm  \
// RUN:   -fmodule-file=%t/header.pcm \
// RUN:   -verify -fsyntax-only

//--- enum.h
enum E { Value };

//--- header.h
#include "enum.h"

//--- A.cppm
module;
#include "enum.h"
export module A;

auto e = Value;

//--- B.cppm
export module B;
import "header.h";

auto e = Value;

//--- use.cpp
// expected-no-diagnostics
import A;
import B;
#include "enum.h"

auto e = Value;
