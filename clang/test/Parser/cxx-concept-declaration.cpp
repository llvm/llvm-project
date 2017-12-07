
// Support parsing of concepts
// Disabled for now.
// expected-no-diagnostics

// RUN:  %clang_cc1 -std=c++14 -fconcepts-ts -x c++ -verify %s
// template<typename T> concept C1 = true;
