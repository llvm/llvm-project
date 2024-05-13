// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: cd %t
//
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -fmodule-name=c \
// RUN:     -fmodule-map-file=c.cppmap -xc++ c.cppmap -emit-module -o c.pcm
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -fmodule-name=a \
// RUN:     -fmodule-map-file=a.cppmap -fmodule-map-file=c.cppmap -xc++ a.cppmap \
// RUN:     -emit-module -o a.pcm
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -fmodule-name=b \
// RUN:     -fmodule-map-file=b.cppmap -fmodule-map-file=c.cppmap -xc++ b.cppmap \
// RUN:     -emit-module -o b.pcm
// RUN: %clang_cc1 -fmodules -fno-implicit-modules -fmodule-name=test \
// RUN:     -fmodule-map-file=test.cppmap -fmodule-map-file=a.cppmap \
// RUN:     -fmodule-map-file=b.cppmap -fmodule-file=a.pcm -fmodule-file=b.pcm -xc++ \
// RUN:     test.cc -emit-llvm -o - | FileCheck test.cc

//--- a.cppmap
module "a" {
  export *
  module "a.h" {
    export *
    header "a.h"
  }
  use "c"
}

//--- b.cppmap
module "b" {
  export *
  module "b.h" {
    export *
    header "b.h"
  }
  use "c"
}

//--- c.cppmap
module "c" {
  export *
  module "c1.h" {
    export *
    textual header "c1.h"
  }
  module "c2.h" {
    export *
    textual header "c2.h"
  }
  module "c3.h" {
    export *
    textual header "c3.h"
  }
}

//--- test.cppmap
module "test" {
  export *
  use "a"
  use "b"
}

//--- a.h
#ifndef A_H_
#define A_H_

#include "c1.h"

namespace q {
template <typename T,
          typename std::enable_if<::p::P<T>::value>::type>
class X {};
}  // namespace q

#include "c3.h"

#endif  // A_H_

//--- b.h
#ifndef B_H_
#define B_H_

#include "c2.h"

#endif  // B_H_

//--- c1.h
#ifndef C1_H_
#define C1_H_

namespace std {
template <class _Tp, _Tp __v>
struct integral_constant {
  static constexpr const _Tp value = __v;
  typedef _Tp value_type;
  typedef integral_constant type;
  constexpr operator value_type() const noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; }
};

template <class _Tp, _Tp __v>
constexpr const _Tp integral_constant<_Tp, __v>::value;

typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;

template <bool, class _Tp = void>
struct enable_if {};
template <class _Tp>
struct enable_if<true, _Tp> {
  typedef _Tp type;
};
}  // namespace std

namespace p {
template <typename T>
struct P : ::std::false_type {};
}

#endif  // C1_H_

//--- c2.h
#ifndef C2_H_
#define C2_H_

#include "c3.h"

enum E {};
namespace p {
template <>
struct P<E> : std::true_type {};
}  // namespace proto2

inline void f(::util::EnumErrorSpace<E>) {}

#endif  // C2_H_

//--- c3.h
#ifndef C3_H_
#define C3_H_

#include "c1.h"

namespace util {

template <typename T>
class ErrorSpaceImpl;

class ErrorSpace {
 protected:
  template <bool* addr>
  struct OdrUse {
    constexpr OdrUse() : b(*addr) {}
    bool& b;
  };
  template <typename T>
  struct Registerer {
    static bool register_token;
    static constexpr OdrUse<&register_token> kRegisterTokenUse{};
  };

 private:
  template <typename T>
  static const ErrorSpace* GetBase() {
    return 0;
  }

  static bool Register(const ErrorSpace* (*space)()) { return true; }
};

template <typename T>
bool ErrorSpace::Registerer<T>::register_token =
    Register(&ErrorSpace::GetBase<T>);

template <typename T>
class ErrorSpaceImpl : public ErrorSpace {
 private:
  static constexpr Registerer<ErrorSpaceImpl> kRegisterer{};
};

template <typename T, typename = typename std::enable_if<p::P<T>::value>::type>
class EnumErrorSpace : public ErrorSpaceImpl<EnumErrorSpace<T>> {};

}  // namespace util
#endif // C3_H_

//--- test.cc
#include "a.h"
#include "b.h"

int main(int, char**) {}

// CHECK-NOT: error
