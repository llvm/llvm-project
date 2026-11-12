// Without PCH
// RUN: %clang_cc1 -fsyntax-only -verify -include %s -include %s %s

// With PCH
// RUN: %clang_cc1 -fsyntax-only -verify %s -chain-include %s -chain-include %s

#ifndef HEADER1
#define HEADER1
//===----------------------------------------------------------------------===//

namespace NS {

template <class _Tp, _Tp __v>
struct TS
{
  static const _Tp value = __v;
};

template <class _Tp, _Tp __v>
const _Tp TS<_Tp, __v>::value;

TS<int, 2> g1;

}

//===----------------------------------------------------------------------===//
#elif not defined(HEADER2)
#define HEADER2
#if !defined(HEADER1)
#error Header inclusion order messed up
#endif

int g2 = NS::TS<int, 2>::value;

//===----------------------------------------------------------------------===//
#else
//===----------------------------------------------------------------------===//

// expected-warning@+1 {{reached main file}}
#warning reached main file

int g3 = NS::TS<int, 2>::value;

//===----------------------------------------------------------------------===//
#endif
