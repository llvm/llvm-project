// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/mod1.cppm -emit-module-interface -o %t/mod1.pcm
// RUN: %clang_cc1 -std=c++20 %t/mod2.cppm -emit-module-interface -o %t/mod2.pcm
// RUN: %clang_cc1 -std=c++20 %t/test.cpp -fprebuilt-module-path=%t -verify -fsyntax-only

//--- size_t.h

extern "C" {
    typedef unsigned int size_t;
}

//--- csize_t
namespace std {
            using :: size_t;
}

//--- align.h
namespace std {
    enum class align_val_t : size_t {};
}

//--- mod1.cppm
module;
#include "size_t.h"
#include "align.h"
export module mod1;
namespace std {
export using std::align_val_t;
}

//--- mod2.cppm
module;
#include "size_t.h"
#include "csize_t"
#include "align.h"
export module mod2;
namespace std {
export using std::align_val_t;
}

//--- test.cpp
// expected-no-diagnostics
import mod1;
import mod2;
void test() {
    std::align_val_t v;
}

