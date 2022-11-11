// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -xc++ -std=c++20 -fmodules -fmodule-name=library \
// RUN:     -emit-module %t/modules.map \
// RUN:     -o %t/module.pcm \
// RUN:     -verify
//
//--- modules.map
module "library" {
	export *
	module "concepts" {
		export *
		header "concepts.h"
	}
	module "conflicting" {
		export *
		header "conflicting.h"
	}
}

//--- concepts.h
#ifndef CONCEPTS_H_
#define CONCEPTS_H_

template <class T>
concept ConflictingConcept = true;

template <class T, class U>
concept same_as = __is_same(T, U);

template<class T> concept truec = true;

int var;

#endif // SAMEAS_CONCEPTS_H

//--- conflicting.h
#ifndef CONFLICTING_H
#define CONFLICTING_H

#include "concepts.h"

template <class T, class U = int>
concept ConflictingConcept = true; // expected-error {{redefinition of concept 'ConflictingConcept' with different template}}
                                   // expected-note@* {{previous definition}}

int same_as; // expected-error {{redefinition of 'same_as' as different kind of symbol}}
             // expected-note@* {{previous definition}}

template<class T> concept var = false; // expected-error {{redefinition of 'var' as different kind of symbol}}
                                       // expected-note@* {{previous definition}}

template<class T> concept truec = true; // expected-error {{redefinition of 'truec'}}
                                        // expected-note@* {{previous definition}}
#endif // CONFLICTING_H
