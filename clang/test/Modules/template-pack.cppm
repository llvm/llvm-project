// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/a.cppm -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -fprebuilt-module-path=%t -fsyntax-only -verify

//--- foo.h

namespace std
{
    template<class _Dom1>
    void operator &&(_Dom1 __v, _Dom1 __w)
    { 
        return;
    }
}

//--- bar.h
namespace std 
{
  template<typename... _Types>
    struct _Traits
    {
      static constexpr bool _S_copy_ctor =
   (__is_trivial(_Types) && ...);
    };

  template<typename... _Types>
    struct variant
    {
      void
      swap(variant& __rhs)
      noexcept((__is_trivial(_Types) && ...))
      {
      }
    };
}

//--- a.cppm
module;
#include "foo.h"
#include "bar.h"
export module a;

//--- b.cppm
// expected-no-diagnostics
module;
#include "bar.h"
export module b;
import a;
