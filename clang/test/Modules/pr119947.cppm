// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fprebuilt-module-path=%t -emit-llvm -o -


//--- a.cppm
export module a;

struct a_inner {
	~a_inner() {
	}
	void f(auto) {
	}
};

export template<typename T>
struct a {
	a() {
		struct local {};
		inner.f(local());
	}
private:
	a_inner inner;
};


namespace {

struct s {
};

} // namespace

void f() {
	a<s> x;
}

//--- use.cpp
import a;

namespace {

struct s {
};

} // namespace

void g() {
	a<s> x;
}

