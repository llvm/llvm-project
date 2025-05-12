// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang -std=c++20 -x c++-module --precompile -c %t/a.cpp -o %t/a.pcm
// RUN: %clang -std=c++20 -fmodule-file=a=%t/a.pcm --precompile -x c++-module -c %t/b.cpp -o %t/b.pcm
// RUN: %clang -std=c++20 -fmodule-file=a=%t/a.pcm -fmodule-file=b=%t/b.pcm -x c++-module --precompile -c %t/c.cpp -o /dev/null

//--- a.cpp
export module a;

struct base {
	base(int) {}
};

export struct a : base {
	using base::base;
};

//--- b.cpp
export module b;

import a;

a b() {
	return a(1);
}

//--- c.cpp
export module c;

import a;
import b;

struct noncopyable {
	noncopyable(noncopyable const &) = delete;
};

struct c {
	noncopyable c0;
	a c1;
};