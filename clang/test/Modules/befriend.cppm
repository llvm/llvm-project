// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cpp -fmodule-file=a=%t/a.pcm -emit-llvm -o /dev/null -verify

//--- a.cppm
module;

namespace n
{

template<typename>
struct a {
	template<typename T>
	friend void aa(a<T>);
};

template<typename T>
inline void aa(a<T>) {
}

} //namespace n

export module a;

namespace n {

export using n::a;
export using n::aa;

}

//--- b.cpp
// expected-no-diagnostics
import a;

void b() {
	aa(n::a<int>());
}
