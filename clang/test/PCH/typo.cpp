
// In header: expected-note{{'boost::function' declared here}}


// In header: expected-note{{'boost::graph::adjacency_list' declared here}}



adjacent_list<int, int> g; // expected-error{{no template named 'adjacent_list'; did you mean 'boost::graph::adjacency_list'?}}
Function<int(int)> f; // expected-error{{no template named 'Function'; did you mean 'boost::function'?}}

// Without PCH
// RUN: %clang_cc1 -include %S/Inputs/typo.hpp -verify %s

// With PCH
// RUN: %clang_cc1 -x c++-header -emit-pch -o %t %S/Inputs/typo.hpp
// RUN: %clang_cc1 -include-pch %t -verify %s
