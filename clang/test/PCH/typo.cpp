// Without PCH
// RUN: %clang_cc1 -include %S/Inputs/typo.hpp -verify %s

// With PCH
// RUN: %clang_cc1 -x c++-header -emit-pch -o %t %S/Inputs/typo.hpp
// RUN: %clang_cc1 -include-pch %t -verify %s

adjacent_list<int, int> g;
// expected-error@-1{{no template named 'adjacent_list'; did you mean 'boost::graph::adjacency_list'?}}
// expected-note@Inputs/typo.hpp:5{{'boost::graph::adjacency_list' declared here}}

Function<int(int)> f;
// expected-error@-1{{no template named 'Function'; did you mean 'boost::function'?}}
// expected-note@Inputs/typo.hpp:2{{'boost::function' declared here}}
