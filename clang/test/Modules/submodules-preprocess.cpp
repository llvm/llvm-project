// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -x objective-c++ -Eonly -fmodule-cache-path %t -I %S/Inputs/submodules %s -verify
// FIXME: When we have a syntax for modules in C++, use that.

@__experimental_modules_import std.vector;

#ifndef HAVE_VECTOR
#  error HAVE_VECTOR macro is not available (but should be)
#endif

#ifdef HAVE_TYPE_TRAITS
#  error HAVE_TYPE_TRAITS_MAP macro is available (but shouldn't be)
#endif

#ifdef HAVE_HASH_MAP
#  error HAVE_HASH_MAP macro is available (but shouldn't be)
#endif

@__experimental_modules_import std.typetraits; // expected-error{{no submodule named 'typetraits' in module 'std'; did you mean 'type_traits'?}}

#ifndef HAVE_VECTOR
#  error HAVE_VECTOR macro is not available (but should be)
#endif

#ifndef HAVE_TYPE_TRAITS
#  error HAVE_TYPE_TRAITS_MAP macro is not available (but should be)
#endif

#ifdef HAVE_HASH_MAP
#  error HAVE_HASH_MAP macro is available (but shouldn't be)
#endif

@__experimental_modules_import std.vector.compare; // expected-error{{no submodule named 'compare' in module 'std.vector'}}

@__experimental_modules_import std; // import everything in 'std'

#ifndef HAVE_VECTOR
#  error HAVE_VECTOR macro is not available (but should be)
#endif

#ifndef HAVE_TYPE_TRAITS
#  error HAVE_TYPE_TRAITS_MAP macro is not available (but should be)
#endif

#ifdef HAVE_HASH_MAP
#  error HAVE_HASH_MAP macro is available (but shouldn't be)
#endif

@__experimental_modules_import std.hash_map;

#ifndef HAVE_VECTOR
#  error HAVE_VECTOR macro is not available (but should be)
#endif

#ifndef HAVE_TYPE_TRAITS
#  error HAVE_TYPE_TRAITS_MAP macro is not available (but should be)
#endif

#ifndef HAVE_HASH_MAP
#  error HAVE_HASH_MAP macro is not available (but should be)
#endif
