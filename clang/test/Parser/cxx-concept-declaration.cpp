
// Support parsing of concepts
// Disabled for now.

// RUN:  %clang_cc1 -std=c++20 -x c++ -verify %s
template<typename T> concept C1 = true;

template<class T>
concept C = true;

template<class T>
class C<int> {}; //expected-error{{identifier followed by '<' indicates a class template specialization but 'C' refers to a concept}}
