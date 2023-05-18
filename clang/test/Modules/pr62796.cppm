// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/Cache.cppm -o %t/Cache.pcm
// RUN: %clang_cc1 -std=c++20 %t/Use.cpp -fmodule-file=Fibonacci.Cache=%t/Cache.pcm \
// RUN:     -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/Cache.cppm -o %t/Cache.pcm
// RUN: %clang_cc1 -std=c++20 %t/Use.cpp -fmodule-file=Fibonacci.Cache=%t/Cache.pcm \
// RUN:     -fsyntax-only -verify

//--- Cache.cppm
export module Fibonacci.Cache;

export namespace Fibonacci
{
	constexpr unsigned long Recursive(unsigned long n)
	{
		if (n == 0)
			return 0;
		if (n == 1)
			return 1;
		return Recursive(n - 2) + Recursive(n - 1);
	}

	template<unsigned long N>
	struct Number{};

	struct DefaultStrategy
	{
		constexpr unsigned long operator()(unsigned long n, auto... other) const
		{
			return (n + ... + other);
		}
	};

    constexpr unsigned long Compute(Number<10ul>, auto strategy)
	{
		return strategy(Recursive(10ul));
	}

	template<unsigned long N, typename Strategy = DefaultStrategy>
	constexpr unsigned long Cache = Compute(Number<N>{}, Strategy{});

    template constexpr unsigned long Cache<10ul>;
}

//--- Use.cpp
// expected-no-diagnostics
import Fibonacci.Cache;

constexpr bool value = Fibonacci::Cache<10ul> == 55;

static_assert(value == true, "");
