// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -fmodule-file=a=%t/a.pcm  -emit-module-interface \
// RUN:     -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 %t/c.cppm -fmodule-file=a=%t/a.pcm -fmodule-file=b=%t/b.pcm \
// RUN:     -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 -DEXPORT_OPERATOR %t/a.cppm -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -DEXPORT_OPERATOR %t/b.cppm -fmodule-file=a=%t/a.pcm  \
// RUN:     -emit-module-interface -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 -DEXPORT_OPERATOR %t/c.cppm -fmodule-file=a=%t/a.pcm \
// RUN:     -fmodule-file=b=%t/b.pcm -fsyntax-only -verify

// Test again with reduced BMI.
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -fmodule-file=a=%t/a.pcm  -emit-reduced-module-interface \
// RUN:     -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 %t/c.cppm -fmodule-file=a=%t/a.pcm -fmodule-file=b=%t/b.pcm \
// RUN:     -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 -DEXPORT_OPERATOR %t/a.cppm -emit-reduced-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -DEXPORT_OPERATOR %t/b.cppm -fmodule-file=a=%t/a.pcm  \
// RUN:     -emit-reduced-module-interface -o %t/b.pcm
// RUN: %clang_cc1 -std=c++20 -DEXPORT_OPERATOR %t/c.cppm -fmodule-file=a=%t/a.pcm \
// RUN:     -fmodule-file=b=%t/b.pcm -fsyntax-only -verify

//--- foo.h
namespace n {

struct s { };

void operator+(s, int) {}

} // namespace n

//--- a.cppm
module;
#include "foo.h"
export module a;
export namespace n {
    using n::s;
#ifdef EXPORT_OPERATOR
    using n::operator+;
#endif
}

//--- b.cppm
export module b;
export import a;

export template<typename T>
void b(T x) {
	n::s() + x;
}

//--- c.cppm
#ifdef EXPORT_OPERATOR
// expected-no-diagnostics
#endif
export module c;
import b;

void c(int x) {
#ifndef EXPORT_OPERATOR
	// expected-error@b.cppm:6 {{invalid operands to binary expression ('n::s' and 'int')}}
    // expected-note@+2 {{in instantiation of function template specialization 'b<int>' requested here}}
#endif
    b(0);

#ifndef EXPORT_OPERATOR
    // expected-error@+2 {{invalid operands to binary expression ('n::s' and 'int')}}
#endif
    n::s() + x;
}
