// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-module-interface -o %t/b.pcm \
// RUN:     -fmodule-file=a=%t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cc -fmodule-file=a=%t/a.pcm -fmodule-file=b=%t/b.pcm \
// RUN:     -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -emit-reduced-module-interface -o %t/b.pcm \
// RUN:     -fmodule-file=a=%t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cc -fmodule-file=a=%t/a.pcm -fmodule-file=b=%t/b.pcm \
// RUN:     -fsyntax-only -verify

//--- a.cppm
export module a;

namespace n {
}

//--- b.cppm
export module b;
import a;

namespace n {
struct monostate {
	friend bool operator==(monostate, monostate) = default;
};

export struct wrapper {
	friend bool operator==(wrapper const &, wrapper const &) = default;

	monostate m_value;
};
}

//--- use.cc
// expected-no-diagnostics
import b;

static_assert(n::wrapper() == n::wrapper());
