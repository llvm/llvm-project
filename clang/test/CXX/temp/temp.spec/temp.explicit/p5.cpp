// RUN: %clang_cc1 -fsyntax-only -verify -Wc++11-compat %s

namespace N {
  template<class T> class Y { // expected-note{{explicit instantiation refers here}}
    void mf() { } 
  };
}

template class Z<int>; // expected-error{{explicit instantiation of non-template class 'Z'}}

// FIXME: This example from the standard is wrong; note posted to CWG reflector
// on 10/27/2009
using N::Y; 
template class Y<int>; // expected-warning{{must occur in}}

template class N::Y<char*>; 
template void N::Y<double>::mf();
