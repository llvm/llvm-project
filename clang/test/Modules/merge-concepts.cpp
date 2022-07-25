// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -xc++ -std=c++20 -fmodules -fmodule-name=library \
// RUN:     -emit-module %t/modules.map \
// RUN:     -o %t/module.pcm
//
//
// RUN: %clang_cc1 -xc++ -std=c++20 -fmodules -fmodule-file=%t/module.pcm  \
// RUN:     -fmodule-map-file=%t/modules.map \
// RUN:     -fsyntax-only -verify %t/use.cpp
//
//--- use.cpp
// expected-no-diagnostics

#include "concepts.h"
#include "format.h"

template <class T> void foo()
  requires same_as<T, T>
{}

//--- modules.map
module "library" {
	export *
	module "concepts" {
		export *
		header "concepts.h"
	}
	module "format" {
		export *
		header "format.h"
	}
}

//--- concepts.h
#ifndef SAMEAS_CONCEPTS_H_
#define SAMEAS_CONCEPTS_H_

#include "same_as.h"

#endif // SAMEAS_CONCEPTS_H

//--- same_as.h
#ifndef SAME_AS_H
#define SAME_AS_H

template <class T, class U>
concept same_as = __is_same(T, U);

#endif // SAME_AS_H

//--- format.h
#ifndef FORMAT_H
#define FORMAT_H

#include "concepts.h"
#include "same_as.h"

template <class T> void foo()
  requires same_as<T, int>
{}

#endif // FORMAT_H