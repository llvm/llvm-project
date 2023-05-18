// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c++ -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -I %S/Inputs/submodules %s -verify
// FIXME: When we have a syntax for modules in C++, use that.

@import std.vector;

vector<int> vi;

// Note: remove_reference is not visible yet.
remove_reference<int&>::type *int_ptr = 0; // expected-error{{declaration of 'remove_reference' must be imported from module 'std.type_traits' before it is required}}
// expected-note@Inputs/submodules/type_traits.h:2{{not visible}}
// expected-note@Inputs/submodules/hash_map.h:1{{not visible}}

@import std.typetraits; // expected-error{{no submodule named 'typetraits' in module 'std'; did you mean 'type_traits'?}}

vector<float> vf;
remove_reference<int&>::type *int_ptr2 = 0;

@import std.vector.compare; // expected-error{{no submodule named 'compare' in module 'std.vector'}}

@import std; // import everything in 'std'

// hash_map still isn't available.
hash_map<int, float> ints_to_floats; // expected-error{{declaration of 'hash_map' must be imported from module 'std.hash_map' before it is required}}

@import std.hash_map;

hash_map<int, float> ints_to_floats2;
