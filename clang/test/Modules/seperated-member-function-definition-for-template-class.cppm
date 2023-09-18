// This comes from the issue report of MSVC
// (https://developercommunity.visualstudio.com/t/c20-modules-unresolved-external-symbol/10049210).
//
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/base.cppm -emit-module-interface -o %t/package-base.pcm
// RUN: %clang_cc1 -std=c++20 %t/child.cppm -emit-module-interface -o %t/package-child.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/package.cppm -emit-module-interface -o %t/package.pcm \
// RUN:     -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fsyntax-only -verify -fprebuilt-module-path=%t

//--- base.cppm
export module package:base;

export struct child;

export
template<class> struct base
{
	child getChild();
};


//--- child.cppm
export module package:child;

import :base;

export struct child : base<void> {};

template<class T>
child base<T>::getChild() { return {}; }

//--- package.cppm
export module package;

export import :base;
export import :child;

//--- use.cpp
// expected-no-diagnostics
import package;

int use()
{
	base<void>{}.getChild();
	base<int>{}.getChild();
	return 0;
}
