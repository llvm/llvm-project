// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -fprebuilt-module-path=%t -emit-llvm -o %t/B.ll

//--- A.cppm
export module A;

export template<typename>
struct holder {
};

struct foo {};

export struct a {
	holder<foo> m;
};

//--- B.cppm
// expected-no-diagnostics
export module B;

import A;

struct foo {};

struct b {
	holder<foo> m;
};