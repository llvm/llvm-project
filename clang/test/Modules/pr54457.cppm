// https://github.com/llvm/llvm-project/issues/54457
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -verify -S -o -
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -verify -S -o -
// RUN: %clang_cc1 -std=c++20 %t/C.cppm -emit-module-interface -o %t/C.pcm
// RUN: %clang_cc1 -std=c++20 %t/UseC.cppm -fprebuilt-module-path=%t -verify -S -o -

//--- A.cppm
// expected-no-diagnostics
export module A;

export template<typename T>
struct s {
	friend s f(s) {
		return s();
	}
};

void g() {
	f(s<int>());
}

//--- B.cppm
// expected-no-diagnostics
export module B;

export template<typename T>
struct s {
	friend constexpr auto f(s) -> s {
		return s();
	}
};

void g() {
	constexpr auto first = f(s<int>());
}

//--- C.cppm
// expected-no-diagnostics
export module C;

export template<typename StandardCharT, int N>
struct basic_symbol_text {
  template<int N2>
  constexpr friend basic_symbol_text operator+(
    const basic_symbol_text&, const basic_symbol_text<char, N2>&) noexcept
  {
    return basic_symbol_text{};
  }
};

constexpr auto xxx = basic_symbol_text<char, 1>{} + basic_symbol_text<char, 1>{};

//--- UseC.cppm
// expected-no-diagnostics
import C;
void foo() {}
