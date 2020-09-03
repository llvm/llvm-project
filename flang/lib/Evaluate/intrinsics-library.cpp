//===-- lib/Evaluate/intrinsics-library.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file defines host runtime functions that can be used for folding
// intrinsic functions.
// The default HostIntrinsicProceduresLibrary is built with <cmath> and
// <complex> functions that are guaranteed to exist from the C++ standard.

#include "intrinsics-library-templates.h"
#include <cmath>
#include <complex>

namespace Fortran::evaluate {

// Note: argument passing is ignored in equivalence
bool HostIntrinsicProceduresLibrary::HasEquivalentProcedure(
    const IntrinsicProcedureRuntimeDescription &sym) const {
  const auto rteProcRange{procedures_.equal_range(sym.name)};
  const size_t nargs{sym.argumentsType.size()};
  for (auto iter{rteProcRange.first}; iter != rteProcRange.second; ++iter) {
    if (nargs == iter->second.argumentsType.size() &&
        sym.returnType == iter->second.returnType &&
        (sym.isElemental || iter->second.isElemental)) {
      bool match{true};
      int pos{0};
      for (const auto &type : sym.argumentsType) {
        if (type != iter->second.argumentsType[pos++]) {
          match = false;
          break;
        }
      }
      if (match) {
        return true;
      }
    }
  }
  return false;
}

// Map numerical intrinsic to  <cmath>/<complex> functions

// Define which host runtime functions will be used for folding

template <typename HostT>
static void AddLibmRealHostProcedures(
    HostIntrinsicProceduresLibrary &hostIntrinsicLibrary) {
  using F = FuncPointer<HostT, HostT>;
  using F2 = FuncPointer<HostT, HostT, HostT>;
  HostRuntimeIntrinsicProcedure libmSymbols[]{
      {"acos", F{std::acos}, true},
      {"acosh", F{std::acosh}, true},
      {"asin", F{std::asin}, true},
      {"asinh", F{std::asinh}, true},
      {"atan", F{std::atan}, true},
      {"atan2", F2{std::atan2}, true},
      {"atanh", F{std::atanh}, true},
      {"cos", F{std::cos}, true},
      {"cosh", F{std::cosh}, true},
      {"erf", F{std::erf}, true},
      {"erfc", F{std::erfc}, true},
      {"exp", F{std::exp}, true},
      {"gamma", F{std::tgamma}, true},
      {"hypot", F2{std::hypot}, true},
      {"log", F{std::log}, true},
      {"log10", F{std::log10}, true},
      {"log_gamma", F{std::lgamma}, true},
      {"mod", F2{std::fmod}, true},
      {"pow", F2{std::pow}, true},
      {"sin", F{std::sin}, true},
      {"sinh", F{std::sinh}, true},
      {"sqrt", F{std::sqrt}, true},
      {"tan", F{std::tan}, true},
      {"tanh", F{std::tanh}, true},
  };
  // Note: cmath does not have modulo and erfc_scaled equivalent

  // Note regarding  lack of bessel function support:
  // C++17 defined standard Bessel math functions std::cyl_bessel_j
  // and std::cyl_neumann that can be used for Fortran j and y
  // bessel functions. However, they are not yet implemented in
  // clang libc++ (ok in GNU libstdc++). C maths functions j0...
  // are not C standard but a GNU extension so they are not used
  // to avoid introducing incompatibilities.
  // Use libpgmath to get bessel function folding support.
  // TODO:  Add Bessel functions when possible.

  for (auto sym : libmSymbols) {
    if (!hostIntrinsicLibrary.HasEquivalentProcedure(sym)) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  }
}

template <typename HostT>
static void AddLibmComplexHostProcedures(
    HostIntrinsicProceduresLibrary &hostIntrinsicLibrary) {
  using F = FuncPointer<std::complex<HostT>, const std::complex<HostT> &>;
  using F2 = FuncPointer<std::complex<HostT>, const std::complex<HostT> &,
      const std::complex<HostT> &>;
  using F2a = FuncPointer<std::complex<HostT>, const HostT &,
      const std::complex<HostT> &>;
  using F2b = FuncPointer<std::complex<HostT>, const std::complex<HostT> &,
      const HostT &>;
  HostRuntimeIntrinsicProcedure libmSymbols[]{
      {"abs", FuncPointer<HostT, const std::complex<HostT> &>{std::abs}, true},
      {"acos", F{std::acos}, true},
      {"acosh", F{std::acosh}, true},
      {"asin", F{std::asin}, true},
      {"asinh", F{std::asinh}, true},
      {"atan", F{std::atan}, true},
      {"atanh", F{std::atanh}, true},
      {"cos", F{std::cos}, true},
      {"cosh", F{std::cosh}, true},
      {"exp", F{std::exp}, true},
      {"log", F{std::log}, true},
      {"pow", F2{std::pow}, true},
      {"pow", F2a{std::pow}, true},
      {"pow", F2b{std::pow}, true},
      {"sin", F{std::sin}, true},
      {"sinh", F{std::sinh}, true},
      {"sqrt", F{std::sqrt}, true},
      {"tan", F{std::tan}, true},
      {"tanh", F{std::tanh}, true},
  };

  for (auto sym : libmSymbols) {
    if (!hostIntrinsicLibrary.HasEquivalentProcedure(sym)) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  }
}

[[maybe_unused]] static void InitHostIntrinsicLibraryWithLibm(
    HostIntrinsicProceduresLibrary &lib) {
  if constexpr (host::FortranTypeExists<float>()) {
    AddLibmRealHostProcedures<float>(lib);
  }
  if constexpr (host::FortranTypeExists<double>()) {
    AddLibmRealHostProcedures<double>(lib);
  }
  if constexpr (host::FortranTypeExists<long double>()) {
    AddLibmRealHostProcedures<long double>(lib);
  }

  if constexpr (host::FortranTypeExists<std::complex<float>>()) {
    AddLibmComplexHostProcedures<float>(lib);
  }
  if constexpr (host::FortranTypeExists<std::complex<double>>()) {
    AddLibmComplexHostProcedures<double>(lib);
  }
  if constexpr (host::FortranTypeExists<std::complex<long double>>()) {
    AddLibmComplexHostProcedures<long double>(lib);
  }
}

#if LINK_WITH_LIBPGMATH
// Only use libpgmath for folding if it is available.
// First declare all libpgmaths functions
#define PGMATH_DECLARE
#include "../runtime/pgmath.h.inc"

// Library versions: P for Precise, R for Relaxed, F for Fast
enum class L { F, R, P };

// Fill the function map used for folding with libpgmath symbols
template <L Lib>
static void AddLibpgmathFloatHostProcedures(
    HostIntrinsicProceduresLibrary &hostIntrinsicLibrary) {
  if constexpr (Lib == L::F) {
    HostRuntimeIntrinsicProcedure pgmathSymbols[]{
#define PGMATH_FAST
#define PGMATH_USE_S(name, function) {#name, function, true},
#include "../runtime/pgmath.h.inc"
    };
    for (auto sym : pgmathSymbols) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  } else if constexpr (Lib == L::R) {
    HostRuntimeIntrinsicProcedure pgmathSymbols[]{
#define PGMATH_RELAXED
#define PGMATH_USE_S(name, function) {#name, function, true},
#include "../runtime/pgmath.h.inc"
    };
    for (auto sym : pgmathSymbols) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  } else {
    static_assert(Lib == L::P && "unexpected libpgmath version");
    HostRuntimeIntrinsicProcedure pgmathSymbols[]{
#define PGMATH_PRECISE
#define PGMATH_USE_S(name, function) {#name, function, true},
#include "../runtime/pgmath.h.inc"
    };
    for (auto sym : pgmathSymbols) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  }
}

template <L Lib>
static void AddLibpgmathDoubleHostProcedures(
    HostIntrinsicProceduresLibrary &hostIntrinsicLibrary) {
  if constexpr (Lib == L::F) {
    HostRuntimeIntrinsicProcedure pgmathSymbols[]{
#define PGMATH_FAST
#define PGMATH_USE_D(name, function) {#name, function, true},
#include "../runtime/pgmath.h.inc"
    };
    for (auto sym : pgmathSymbols) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  } else if constexpr (Lib == L::R) {
    HostRuntimeIntrinsicProcedure pgmathSymbols[]{
#define PGMATH_RELAXED
#define PGMATH_USE_D(name, function) {#name, function, true},
#include "../runtime/pgmath.h.inc"
    };
    for (auto sym : pgmathSymbols) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  } else {
    static_assert(Lib == L::P && "unexpected libpgmath version");
    HostRuntimeIntrinsicProcedure pgmathSymbols[]{
#define PGMATH_PRECISE
#define PGMATH_USE_D(name, function) {#name, function, true},
#include "../runtime/pgmath.h.inc"
    };
    for (auto sym : pgmathSymbols) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  }
}

// Note: Lipgmath uses _Complex but the front-end use std::complex for folding.
// std::complex and _Complex are layout compatible but are not guaranteed
// to be linkage compatible. For instance, on i386, float _Complex is returned
// by a pair of register but std::complex<float> is returned by structure
// address. To fix the issue, wrapper around C _Complex functions are defined
// below.

template <typename T> struct ToStdComplex {
  using Type = T;
  using AType = Type;
};

template <typename F, F func> struct CComplexFunc {};
template <typename R, typename... A, FuncPointer<R, A...> func>
struct CComplexFunc<FuncPointer<R, A...>, func> {
  static typename ToStdComplex<R>::Type wrapper(
      typename ToStdComplex<A>::AType... args) {
    R res{func(*reinterpret_cast<A *>(&args)...)};
    return *reinterpret_cast<typename ToStdComplex<R>::Type *>(&res);
  }
};

template <L Lib>
static void AddLibpgmathComplexHostProcedures(
    HostIntrinsicProceduresLibrary &hostIntrinsicLibrary) {
  if constexpr (Lib == L::F) {
    HostRuntimeIntrinsicProcedure pgmathSymbols[]{
#define PGMATH_FAST
#define PGMATH_USE_C(name, function) \
  {#name, CComplexFunc<decltype(&function), &function>::wrapper, true},
#include "../runtime/pgmath.h.inc"
    };
    for (auto sym : pgmathSymbols) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  } else if constexpr (Lib == L::R) {
    HostRuntimeIntrinsicProcedure pgmathSymbols[]{
#define PGMATH_RELAXED
#define PGMATH_USE_C(name, function) \
  {#name, CComplexFunc<decltype(&function), &function>::wrapper, true},
#include "../runtime/pgmath.h.inc"
    };
    for (auto sym : pgmathSymbols) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  } else {
    static_assert(Lib == L::P && "unexpected libpgmath version");
    HostRuntimeIntrinsicProcedure pgmathSymbols[]{
#define PGMATH_PRECISE
#define PGMATH_USE_C(name, function) \
  {#name, CComplexFunc<decltype(&function), &function>::wrapper, true},
#include "../runtime/pgmath.h.inc"
    };
    for (auto sym : pgmathSymbols) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  }

  // cmath is used to complement pgmath when symbols are not available
  using HostT = float;
  using CHostT = std::complex<HostT>;
  using CmathF = FuncPointer<CHostT, const CHostT &>;
  hostIntrinsicLibrary.AddProcedure(
      {"abs", FuncPointer<HostT, const CHostT &>{std::abs}, true});
  hostIntrinsicLibrary.AddProcedure({"acosh", CmathF{std::acosh}, true});
  hostIntrinsicLibrary.AddProcedure({"asinh", CmathF{std::asinh}, true});
  hostIntrinsicLibrary.AddProcedure({"atanh", CmathF{std::atanh}, true});
}

template <L Lib>
static void AddLibpgmathDoubleComplexHostProcedures(
    HostIntrinsicProceduresLibrary &hostIntrinsicLibrary) {
  if constexpr (Lib == L::F) {
    HostRuntimeIntrinsicProcedure pgmathSymbols[]{
#define PGMATH_FAST
#define PGMATH_USE_Z(name, function) \
  {#name, CComplexFunc<decltype(&function), &function>::wrapper, true},
#include "../runtime/pgmath.h.inc"
    };
    for (auto sym : pgmathSymbols) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  } else if constexpr (Lib == L::R) {
    HostRuntimeIntrinsicProcedure pgmathSymbols[]{
#define PGMATH_RELAXED
#define PGMATH_USE_Z(name, function) \
  {#name, CComplexFunc<decltype(&function), &function>::wrapper, true},
#include "../runtime/pgmath.h.inc"
    };
    for (auto sym : pgmathSymbols) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  } else {
    static_assert(Lib == L::P && "unexpected libpgmath version");
    HostRuntimeIntrinsicProcedure pgmathSymbols[]{
#define PGMATH_PRECISE
#define PGMATH_USE_Z(name, function) \
  {#name, CComplexFunc<decltype(&function), &function>::wrapper, true},
#include "../runtime/pgmath.h.inc"
    };
    for (auto sym : pgmathSymbols) {
      hostIntrinsicLibrary.AddProcedure(std::move(sym));
    }
  }

  // cmath is used to complement pgmath when symbols are not available
  using HostT = double;
  using CHostT = std::complex<HostT>;
  using CmathF = FuncPointer<CHostT, const CHostT &>;
  hostIntrinsicLibrary.AddProcedure(
      {"abs", FuncPointer<HostT, const CHostT &>{std::abs}, true});
  hostIntrinsicLibrary.AddProcedure({"acosh", CmathF{std::acosh}, true});
  hostIntrinsicLibrary.AddProcedure({"asinh", CmathF{std::asinh}, true});
  hostIntrinsicLibrary.AddProcedure({"atanh", CmathF{std::atanh}, true});
}

template <L Lib>
static void InitHostIntrinsicLibraryWithLibpgmath(
    HostIntrinsicProceduresLibrary &lib) {
  if constexpr (host::FortranTypeExists<float>()) {
    AddLibpgmathFloatHostProcedures<Lib>(lib);
  }
  if constexpr (host::FortranTypeExists<double>()) {
    AddLibpgmathDoubleHostProcedures<Lib>(lib);
  }
  if constexpr (host::FortranTypeExists<std::complex<float>>()) {
    AddLibpgmathComplexHostProcedures<Lib>(lib);
  }
  if constexpr (host::FortranTypeExists<std::complex<double>>()) {
    AddLibpgmathDoubleComplexHostProcedures<Lib>(lib);
  }
  // No long double functions in libpgmath
  if constexpr (host::FortranTypeExists<long double>()) {
    AddLibmRealHostProcedures<long double>(lib);
  }
  if constexpr (host::FortranTypeExists<std::complex<long double>>()) {
    AddLibmComplexHostProcedures<long double>(lib);
  }
}
#endif // LINK_WITH_LIBPGMATH

// Define which host runtime functions will be used for folding
HostIntrinsicProceduresLibrary::HostIntrinsicProceduresLibrary() {
  // TODO: When command line options regarding targeted numerical library is
  // available, this needs to be revisited to take it into account. So far,
  // default to libpgmath if F18 is built with it.
#if LINK_WITH_LIBPGMATH
  // This looks and is stupid for now (until TODO above), but it is needed
  // to silence clang warnings on unused symbols if all declared pgmath
  // symbols are not used somewhere.
  if (true) {
    InitHostIntrinsicLibraryWithLibpgmath<L::P>(*this);
  } else if (false) {
    InitHostIntrinsicLibraryWithLibpgmath<L::F>(*this);
  } else {
    InitHostIntrinsicLibraryWithLibpgmath<L::R>(*this);
  }
#else
  InitHostIntrinsicLibraryWithLibm(*this);
#endif
}
} // namespace Fortran::evaluate
