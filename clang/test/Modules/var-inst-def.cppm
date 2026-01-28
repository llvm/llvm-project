// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -fmodule-name=A -xc++ -emit-module -fmodules \
// RUN:   -fno-cxx-modules -fno-implicit-modules \
// RUN:   -fmodule-map-file-home-is-cwd -std=c++20 -I. a.modulemap -o a.pcm
//
// RUN: %clang_cc1 -fmodule-name=B -xc++ -emit-module -fmodules \
// RUN:   -fno-cxx-modules -fno-implicit-modules \
// RUN:   -fmodule-map-file-home-is-cwd -std=c++20 -I. b.modulemap -o b.pcm
//
// RUN: %clang_cc1 -fmodule-name=C -xc++ -emit-module -fmodules \
// RUN:   -fno-cxx-modules -fno-implicit-modules \
// RUN:   -fmodule-map-file-home-is-cwd -std=c++20 -I. c.modulemap -o c.pcm
//
// RUN: %clang_cc1 -fno-cxx-modules -fmodules -fno-implicit-modules \
// RUN:   -fmodule-map-file-home-is-cwd \
// RUN:   -fmodule-file=a.pcm -fmodule-file=b.pcm -fmodule-file=c.pcm \
// RUN:   -std=c++20 -I. main.cpp -o /dev/null

//--- a.modulemap
module "A" { header "a.h" }
//--- b.modulemap
module "B" { header "b.h" }
//--- c.modulemap
module "C" { header "c.h" }

//--- common.h
#pragma once
#include "stl.h"

//--- a.h
#pragma once
#include "common.h"
#include "repro.h"

//--- b.h
#pragma once
#include "common.h"
#include "repro.h"

//--- c.h
#pragma once
#include "common.h"
#include "repro.h"

//--- repro.h
#pragma once
#include "stl.h"

namespace k {
template <template <typename> class , typename >
struct is_instantiation : std::integral_constant<bool, false> {};
template <template <typename> class C, typename T>
constexpr bool is_instantiation_v = is_instantiation<C, T>::value;
}  

struct ThreadState;

namespace cc::subtle {
template <typename T>
class U;
}  
namespace cc {
template <typename T> class Co;
namespace internal {
template <typename T>
class Promise {
  static_assert(!k::is_instantiation_v<subtle::U, T>);
};
}  
}

//--- stl.h
#pragma once
namespace std {
inline namespace abi {
template <class _Tp, _Tp __v>
struct integral_constant {
  static const _Tp value = __v;
};
template <class _Tp, class _Up>
constexpr bool is_same_v = __is_same(_Tp, _Up);
template <class _Tp>
using decay_t = __decay(_Tp);

template <class>
struct __invoke_result_impl ;
template <class... _Args>
using invoke_result_t = __invoke_result_impl<_Args...>;
}
}

//--- main.cpp
#include "stl.h"
#include "a.h"

namespace cc {
template <typename F>
  requires k::is_instantiation_v<Co, std::invoke_result_t<F>>
using result_type =
    std::invoke_result_t<F>;
}  
namespace cc::internal {
class final {
 Promise<ThreadState> outgoing_work_;
};
}
