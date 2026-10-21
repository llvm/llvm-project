// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++23 -emit-module-interface %t/stdx.cc -o %t/stdx.pcm
// RUN: %clang_cc1 -std=c++23 -emit-module-interface %t/a.cppm -o %t/a.pcm -fmodule-file=stdx=%t/stdx.pcm
// RUN: %clang_cc1 -std=c++23 %t/b.cppm -fmodule-file=stdx=%t/stdx.pcm -fmodule-file=a=%t/a.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++23 %t/c.cpp -fmodule-file=stdx=%t/stdx.pcm -fmodule-file=a=%t/a.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++23 %t/d.cpp -fmodule-file=stdx=%t/stdx.pcm -fmodule-file=a=%t/a.pcm -fsyntax-only -verify

// Test again with Reduced BMI
// RUN: %clang_cc1 -std=c++23 -emit-reduced-module-interface %t/stdx.cc -o %t/stdx.pcm
// RUN: %clang_cc1 -std=c++23 -emit-reduced-module-interface %t/a.cppm -o %t/a.pcm -fmodule-file=stdx=%t/stdx.pcm
// RUN: %clang_cc1 -std=c++23 %t/b.cppm -fmodule-file=stdx=%t/stdx.pcm -fmodule-file=a=%t/a.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++23 %t/c.cpp -fmodule-file=stdx=%t/stdx.pcm -fmodule-file=a=%t/a.pcm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++23 %t/d.cpp -fmodule-file=stdx=%t/stdx.pcm -fmodule-file=a=%t/a.pcm -fsyntax-only -verify

//--- pipe.hh
namespace demo {

template <typename Derived>
struct RAC {
  template <typename R, typename Self>
  friend constexpr auto operator|(R&&, Self&&);
};

template <typename A, typename Arg>
struct P : RAC<P<A, Arg>> {
  template <typename R>
  constexpr auto operator()(R&& r) const {
    return A{}(static_cast<R&&>(r));
  }
};

struct myadapt {
  template <class... Args>
  constexpr auto operator()(Args&&...) const {
    return P<myadapt, Args...>{};
  }
};

template <typename R, typename Self>
constexpr auto operator|(R&& r, Self&& self) {
  return static_cast<Self&&>(self)(static_cast<R&&>(r));
}

inline constexpr myadapt adapt{};

} // namespace demo

//--- stdx.cc
module;

#include "pipe.hh"

export module stdx;

export namespace demo {

using demo::myadapt;
using demo::adapt;

// To avoid another GCC-only problem
inline void force() { struct X{}; (void)(X{} | demo::adapt(1)); }

} // namespace demo

//--- a.cppm
export module a;
import stdx;
void g() { (void) demo::adapt('.'); }

//--- b.cppm
// expected-no-diagnostics
export module b;
import stdx;
import a;
struct view {};
void f() { (void)(view{} | demo::adapt('.')); }

//--- c.cpp
// expected-no-diagnostics
import stdx;
import a;
#include "pipe.hh"
struct view {};
void f() { (void)(view{} | demo::adapt('.')); }

//--- d.cpp
// expected-no-diagnostics
#include "pipe.hh"
import stdx;
import a;
struct view {};
void f() { (void)(view{} | demo::adapt('.')); }
