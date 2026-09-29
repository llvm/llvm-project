// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -fmodule-file=a=%t/a.pcm -fsyntax-only -verify

//--- a.cppm
export module a;

namespace n {

struct monostate {
	friend auto operator==(monostate, monostate) -> bool = default;
};

export struct a {
	friend auto operator==(a, a) -> bool = default;
	monostate m;
};

} // namespace n

//--- b.cppm
// expected-no-diagnostics
export module b;

import a;

namespace n {

export auto b() -> bool {
	return a() == a();
}

} // namespace n
