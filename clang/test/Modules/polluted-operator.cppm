// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/a.cppm -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cppm -fprebuilt-module-path=%t -emit-module-interface -o %t/b.pcm -verify

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
// The operator&& defined in 'foo.h' will pollute the 
// expression '__is_trivial(_Types) && ...' in bar.h
#include "foo.h"
#include "bar.h"
export module a;

//--- b.cppm
module;
#include "bar.h"
export module b;
import a;

void b() {
  std::variant<int, double> v;
}

// expected-error@* {{has different definitions in different modules; first difference is defined here found data member '_S_copy_ctor' with an initializer}}
// expected-note@* {{but in 'a.<global>' found data member '_S_copy_ctor' with a different initializer}}
// expected-error@* {{from module 'a.<global>' is not present in definition of 'variant<_Types...>' provided earlier}}
// expected-note@* {{declaration of 'swap' does not match}}

//--- c.cppm
module;
#include "bar.h"
export module c;
import a;

// expected-error@* {{has different definitions in different modules; first difference is defined here found data member '_S_copy_ctor' with an initializer}}
// expected-note@* {{but in 'a.<global>' found data member '_S_copy_ctor' with a different initializer}}
