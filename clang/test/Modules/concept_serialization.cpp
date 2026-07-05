// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++20 -fmodules-cache-path=%t -x c++ %s -verify
// expected-no-diagnostics
#pragma clang module build std
module std [system] { module concepts [system] {} }
#pragma clang module contents

#pragma clang module begin std.concepts
template <class T>
T declval();
template<class T, class U>
concept common_reference_with = T::val;
template<class T>
concept input_or_output_iterator = true;
template <class T>
concept input_iterator = input_or_output_iterator<T> &&
                         common_reference_with<decltype(declval<T&>)&&, T&>;
#pragma clang module end /*std.concepts*/
#pragma clang module endbuild /*std*/

#pragma clang module import std.concepts
template<input_or_output_iterator>
struct iter_value_or_void{};
// ensure that we don't assert on a subsumption check due to improper
// deserialization.
template<input_iterator I>
struct iter_value_or_void<I>{};
