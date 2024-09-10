// From https://github.com/llvm/llvm-project/issues/77953
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/a.cppm -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -fmodule-file=a=%t/a.pcm %t/b.cpp -fsyntax-only -verify

//--- a.cppm
export module a;

template<typename, typename>
concept c = true;

export template<typename... Ts>
struct a {
	template<typename... Us> requires(... and c<Ts, Us>)
	friend bool operator==(a, a<Us...>) {
		return true;
	}
};

template struct a<>;

//--- b.cpp
// expected-no-diagnostics
import a;
template struct a<int>;
