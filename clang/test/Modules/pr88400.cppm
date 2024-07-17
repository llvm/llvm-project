// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/bar.cppm -emit-module-interface -o %t/bar.pcm
// RUN: %clang_cc1 -std=c++20 %t/foo.cc -fmodule-file=bar=%t/bar.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/bar.cc -fmodule-file=bar=%t/bar.pcm -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/bar.cppm -emit-reduced-module-interface -o %t/bar.pcm
// RUN: %clang_cc1 -std=c++20 %t/foo.cc -fmodule-file=bar=%t/bar.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/bar.cc -fmodule-file=bar=%t/bar.pcm -fsyntax-only -verify

//--- header.h
#pragma once

namespace N {
    template<typename T>
    concept X = true;

    template<X T>
    class Y {
    public:
        template<X U>
        friend class Y;
    };

    inline Y<int> x;
}

//--- bar.cppm
module;

#include "header.h"

export module bar;

namespace N {
    // To make sure N::Y won't get elided.
    using N::x;
}

//--- foo.cc
// expected-no-diagnostics
#include "header.h"

import bar;

void y() {
    N::Y<int> y{};
};

//--- bar.cc
// expected-no-diagnostics
import bar;

#include "header.h"

void y() {
    N::Y<int> y{};
};

