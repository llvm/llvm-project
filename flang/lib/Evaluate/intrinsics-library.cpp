//===-- lib/Evaluate/intrinsics-library.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file defines host runtime functions that can be used for folding
// intrinsic functions.
// The default host runtime folders are built with <cmath> and
// <complex> functions that are guaranteed to exist from the C++ standard.

#include "flang/Evaluate/intrinsics-library.h"
#include "fold-implementation.h"
#include "host.h"
#include "flang/Common/static-multimap-view.h"
#include "flang/Evaluate/expression.h"
#include <cfloat>
#include <cmath>
#include <complex>
#include <functional>
#include <type_traits>

namespace Fortran::evaluate {

// Define a vector like class that can hold an arbitrary number of
// Dynamic type and be built at compile time. This is like a
// std::vector<DynamicType>, but constexpr only.
template <typename... FortranType> struct TypeVectorStorage {
  static constexpr DynamicType values[]{FortranType{}.GetType()...};
  static constexpr const DynamicType *start{&values[0]};
  static constexpr const DynamicType *end{start + sizeof...(FortranType)};
};
template <> struct TypeVectorStorage<> {
  static constexpr const DynamicType *start{nullptr}, *end{nullptr};
};
struct TypeVector {
  template <typename... FortranType> static constexpr TypeVector Create() {
    using storage = TypeVectorStorage<FortranType...>;
    return TypeVector{storage::start, storage::end, sizeof...(FortranType)};
  }
  constexpr size_t size() const { return size_; };
  using const_iterator = const DynamicType *;
  constexpr const_iterator begin() const { return startPtr; }
  constexpr const_iterator end() const { return endPtr; }
  const DynamicType &operator[](size_t i) const { return *(startPtr + i); }

  const DynamicType *startPtr{nullptr};
  const DynamicType *endPtr{nullptr};
  const size_t size_;
};
inline bool operator==(
    const TypeVector &lhs, const std::vector<DynamicType> &rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i{0}; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

// HostRuntimeFunction holds a pointer to a Folder function that can fold
// a Fortran scalar intrinsic using host runtime functions (e.g libm).
// The folder take care of all conversions between Fortran types and the related
// host types as well as setting and cleaning-up the floating point environment.
// HostRuntimeFunction are intended to be built at compile time (members are all
// constexpr constructible) so that they can be stored in a compile time static
// map.
struct HostRuntimeFunction {
  using Folder = Expr<SomeType> (*)(
      FoldingContext &, std::vector<Expr<SomeType>> &&);
  using Key = std::string_view;
  // Needed for implicit compare with keys.
  constexpr operator Key() const { return key; }
  // Name of the related Fortran intrinsic.
  Key key;
  // DynamicType of the Expr<SomeType> returns by folder.
  DynamicType resultType;
  // DynamicTypes expected for the Expr<SomeType> arguments of the folder.
  // The folder will crash if provided arguments of different types.
  TypeVector argumentTypes;
  // Folder to be called to fold the intrinsic with host runtime. The provided
  // Expr<SomeType> arguments must wrap scalar constants of the type described
  // in argumentTypes, otherwise folder will crash. Any floating point issue
  // raised while executing the host runtime will be reported in FoldingContext
  // messages.
  Folder folder;
};

// Translate a host function type signature (template arguments) into a
// constexpr data representation based on Fortran DynamicType that can be
// stored.
template <typename TR, typename... TA> using FuncPointer = TR (*)(TA...);
template <typename T> struct FuncTypeAnalyzer {};
template <typename HostTR, typename... HostTA>
struct FuncTypeAnalyzer<FuncPointer<HostTR, HostTA...>> {
  static constexpr DynamicType result{host::FortranType<HostTR>{}.GetType()};
  static constexpr TypeVector arguments{
      TypeVector::Create<host::FortranType<HostTA>...>()};
};

// Define helpers to deal with host floating environment.
template <typename TR>
static void CheckFloatingPointIssues(
    host::HostFloatingPointEnvironment &hostFPE, const Scalar<TR> &x) {
  if constexpr (TR::category == TypeCategory::Complex ||
      TR::category == TypeCategory::Real) {
    if (x.IsNotANumber()) {
      hostFPE.SetFlag(RealFlag::InvalidArgument);
    } else if (x.IsInfinite()) {
      hostFPE.SetFlag(RealFlag::Overflow);
    }
  }
}
// Software Subnormal Flushing helper.
// Only flush floating-points. Forward other scalars untouched.
// Software flushing is only performed if hardware flushing is not available
// because it may not result in the same behavior as hardware flushing.
// Some runtime implementations are "working around" subnormal flushing to
// return results that they deem better than returning the result they would
// with a null argument. An example is logf that should return -inf if arguments
// are flushed to zero, but some implementations return -1.03972076416015625e2_4
// for all subnormal values instead. It is impossible to reproduce this with the
// simple software flushing below.
template <typename T>
static constexpr inline const Scalar<T> FlushSubnormals(Scalar<T> &&x) {
  if constexpr (T::category == TypeCategory::Real ||
      T::category == TypeCategory::Complex) {
    return x.FlushSubnormalToZero();
  }
  return x;
}

// This is the kernel called by all HostRuntimeFunction folders, it convert the
// Fortran Expr<SomeType> to the host runtime function argument types, calls
// the runtime function, and wrap back the result into an Expr<SomeType>.
// It deals with host floating point environment set-up and clean-up.
template <typename FuncType, typename TR, typename... TA, size_t... I>
static Expr<SomeType> ApplyHostFunctionHelper(FuncType func,
    FoldingContext &context, std::vector<Expr<SomeType>> &&args,
    std::index_sequence<I...>) {
  host::HostFloatingPointEnvironment hostFPE;
  hostFPE.SetUpHostFloatingPointEnvironment(context);
  host::HostType<TR> hostResult{};
  Scalar<TR> result{};
  std::tuple<Scalar<TA>...> scalarArgs{
      GetScalarConstantValue<TA>(args[I]).value()...};
  if (context.flushSubnormalsToZero() &&
      !hostFPE.hasSubnormalFlushingHardwareControl()) {
    hostResult = func(host::CastFortranToHost<TA>(
        FlushSubnormals<TA>(std::move(std::get<I>(scalarArgs))))...);
    result = FlushSubnormals<TR>(host::CastHostToFortran<TR>(hostResult));
  } else {
    hostResult = func(host::CastFortranToHost<TA>(std::get<I>(scalarArgs))...);
    result = host::CastHostToFortran<TR>(hostResult);
  }
  if (!hostFPE.hardwareFlagsAreReliable()) {
    CheckFloatingPointIssues<TR>(hostFPE, result);
  }
  hostFPE.CheckAndRestoreFloatingPointEnvironment(context);
  return AsGenericExpr(Constant<TR>(std::move(result)));
}
template <typename HostTR, typename... HostTA>
Expr<SomeType> ApplyHostFunction(FuncPointer<HostTR, HostTA...> func,
    FoldingContext &context, std::vector<Expr<SomeType>> &&args) {
  return ApplyHostFunctionHelper<decltype(func), host::FortranType<HostTR>,
      host::FortranType<HostTA>...>(
      func, context, std::move(args), std::index_sequence_for<HostTA...>{});
}

// FolderFactory builds a HostRuntimeFunction for the host runtime function
// passed as a template argument.
// Its static member function "fold" is the resulting folder. It captures the
// host runtime function pointer and pass it to the host runtime function folder
// kernel.
template <typename HostFuncType, HostFuncType func> class FolderFactory {
public:
  static constexpr HostRuntimeFunction Create(const std::string_view &name) {
    return HostRuntimeFunction{name, FuncTypeAnalyzer<HostFuncType>::result,
        FuncTypeAnalyzer<HostFuncType>::arguments, &Fold};
  }

private:
  static Expr<SomeType> Fold(
      FoldingContext &context, std::vector<Expr<SomeType>> &&args) {
    return ApplyHostFunction(func, context, std::move(args));
  }
};

// Define host runtime libraries that can be used for folding and
// fill their description if they are available.
enum class LibraryVersion {
  Libm,
  LibmExtensions,
  PgmathFast,
  PgmathRelaxed,
  PgmathPrecise
};
template <typename HostT, LibraryVersion> struct HostRuntimeLibrary {
  // When specialized, this class holds a static constexpr table containing
  // all the HostRuntimeLibrary for functions of library LibraryVersion
  // that returns a value of type HostT.
};

using HostRuntimeMap = common::StaticMultimapView<HostRuntimeFunction>;

// Map numerical intrinsic to  <cmath>/<complex> functions
// (Note: ABS() is folded in fold-real.cpp.)
template <typename HostT>
struct HostRuntimeLibrary<HostT, LibraryVersion::Libm> {
  using F = FuncPointer<HostT, HostT>;
  using F2 = FuncPointer<HostT, HostT, HostT>;
  static constexpr HostRuntimeFunction table[]{
      FolderFactory<F, F{std::acos}>::Create("acos"),
      FolderFactory<F, F{std::acosh}>::Create("acosh"),
      FolderFactory<F, F{std::asin}>::Create("asin"),
      FolderFactory<F, F{std::asinh}>::Create("asinh"),
      FolderFactory<F, F{std::atan}>::Create("atan"),
      FolderFactory<F2, F2{std::atan2}>::Create("atan2"),
      FolderFactory<F, F{std::atanh}>::Create("atanh"),
      FolderFactory<F, F{std::cos}>::Create("cos"),
      FolderFactory<F, F{std::cosh}>::Create("cosh"),
      FolderFactory<F, F{std::erf}>::Create("erf"),
      FolderFactory<F, F{std::erfc}>::Create("erfc"),
      FolderFactory<F, F{std::exp}>::Create("exp"),
      FolderFactory<F, F{std::tgamma}>::Create("gamma"),
      FolderFactory<F, F{std::log}>::Create("log"),
      FolderFactory<F, F{std::log10}>::Create("log10"),
      FolderFactory<F, F{std::lgamma}>::Create("log_gamma"),
      FolderFactory<F2, F2{std::pow}>::Create("pow"),
      FolderFactory<F, F{std::sin}>::Create("sin"),
      FolderFactory<F, F{std::sinh}>::Create("sinh"),
      FolderFactory<F, F{std::tan}>::Create("tan"),
      FolderFactory<F, F{std::tanh}>::Create("tanh"),
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
  static constexpr HostRuntimeMap map{table};
  static_assert(map.Verify(), "map must be sorted");
};
template <typename HostT>
struct HostRuntimeLibrary<std::complex<HostT>, LibraryVersion::Libm> {
  using F = FuncPointer<std::complex<HostT>, const std::complex<HostT> &>;
  using F2 = FuncPointer<std::complex<HostT>, const std::complex<HostT> &,
      const std::complex<HostT> &>;
  using F2A = FuncPointer<std::complex<HostT>, const HostT &,
      const std::complex<HostT> &>;
  using F2B = FuncPointer<std::complex<HostT>, const std::complex<HostT> &,
      const HostT &>;
  static constexpr HostRuntimeFunction table[]{
      FolderFactory<F, F{std::acos}>::Create("acos"),
      FolderFactory<F, F{std::acosh}>::Create("acosh"),
      FolderFactory<F, F{std::asin}>::Create("asin"),
      FolderFactory<F, F{std::asinh}>::Create("asinh"),
      FolderFactory<F, F{std::atan}>::Create("atan"),
      FolderFactory<F, F{std::atanh}>::Create("atanh"),
      FolderFactory<F, F{std::cos}>::Create("cos"),
      FolderFactory<F, F{std::cosh}>::Create("cosh"),
      FolderFactory<F, F{std::exp}>::Create("exp"),
      FolderFactory<F, F{std::log}>::Create("log"),
      FolderFactory<F2, F2{std::pow}>::Create("pow"),
      FolderFactory<F2A, F2A{std::pow}>::Create("pow"),
      FolderFactory<F2B, F2B{std::pow}>::Create("pow"),
      FolderFactory<F, F{std::sin}>::Create("sin"),
      FolderFactory<F, F{std::sinh}>::Create("sinh"),
      FolderFactory<F, F{std::sqrt}>::Create("sqrt"),
      FolderFactory<F, F{std::tan}>::Create("tan"),
      FolderFactory<F, F{std::tanh}>::Create("tanh"),
  };
  static constexpr HostRuntimeMap map{table};
  static_assert(map.Verify(), "map must be sorted");
};
// Note regarding cmath:
//  - cmath does not have modulo and erfc_scaled equivalent
//  - C++17 defined standard Bessel math functions std::cyl_bessel_j
//    and std::cyl_neumann that can be used for Fortran j and y
//    bessel functions. However, they are not yet implemented in
//    clang libc++ (ok in GNU libstdc++). Instead, the Posix libm
//    extensions are used when available below.

#if _POSIX_C_SOURCE >= 200112L || _XOPEN_SOURCE >= 600
/// Define libm extensions
/// Bessel functions are defined in POSIX.1-2001.

template <> struct HostRuntimeLibrary<float, LibraryVersion::LibmExtensions> {
  using F = FuncPointer<float, float>;
  using FN = FuncPointer<float, int, float>;
  static constexpr HostRuntimeFunction table[]{
      FolderFactory<F, F{::j0f}>::Create("bessel_j0"),
      FolderFactory<F, F{::j1f}>::Create("bessel_j1"),
      FolderFactory<FN, FN{::jnf}>::Create("bessel_jn"),
      FolderFactory<F, F{::y0f}>::Create("bessel_y0"),
      FolderFactory<F, F{::y1f}>::Create("bessel_y1"),
      FolderFactory<FN, FN{::ynf}>::Create("bessel_yn"),
  };
  static constexpr HostRuntimeMap map{table};
  static_assert(map.Verify(), "map must be sorted");
};

template <> struct HostRuntimeLibrary<double, LibraryVersion::LibmExtensions> {
  using F = FuncPointer<double, double>;
  using FN = FuncPointer<double, int, double>;
  static constexpr HostRuntimeFunction table[]{
      FolderFactory<F, F{::j0}>::Create("bessel_j0"),
      FolderFactory<F, F{::j1}>::Create("bessel_j1"),
      FolderFactory<FN, FN{::jn}>::Create("bessel_jn"),
      FolderFactory<F, F{::y0}>::Create("bessel_y0"),
      FolderFactory<F, F{::y1}>::Create("bessel_y1"),
      FolderFactory<FN, FN{::yn}>::Create("bessel_yn"),
  };
  static constexpr HostRuntimeMap map{table};
  static_assert(map.Verify(), "map must be sorted");
};

#if LDBL_MANT_DIG == 80 || LDBL_MANT_DIG == 113
template <>
struct HostRuntimeLibrary<long double, LibraryVersion::LibmExtensions> {
  using F = FuncPointer<long double, long double>;
  using FN = FuncPointer<long double, int, long double>;
  static constexpr HostRuntimeFunction table[]{
      FolderFactory<F, F{::j0l}>::Create("bessel_j0"),
      FolderFactory<F, F{::j1l}>::Create("bessel_j1"),
      FolderFactory<FN, FN{::jnl}>::Create("bessel_jn"),
      FolderFactory<F, F{::y0l}>::Create("bessel_y0"),
      FolderFactory<F, F{::y1l}>::Create("bessel_y1"),
      FolderFactory<FN, FN{::ynl}>::Create("bessel_yn"),
  };
  static constexpr HostRuntimeMap map{table};
  static_assert(map.Verify(), "map must be sorted");
};
#endif // LDBL_MANT_DIG == 80 || LDBL_MANT_DIG == 113
#endif

/// Define pgmath description
#if LINK_WITH_LIBPGMATH
// Only use libpgmath for folding if it is available.
// First declare all libpgmaths functions
#define PGMATH_LINKING
#define PGMATH_DECLARE
#include "flang/Evaluate/pgmath.h.inc"

#define REAL_FOLDER(name, func) \
  FolderFactory<decltype(&func), &func>::Create(#name)
template <> struct HostRuntimeLibrary<float, LibraryVersion::PgmathFast> {
  static constexpr HostRuntimeFunction table[]{
#define PGMATH_FAST
#define PGMATH_USE_S(name, func) REAL_FOLDER(name, func),
#include "flang/Evaluate/pgmath.h.inc"
  };
  static constexpr HostRuntimeMap map{table};
  static_assert(map.Verify(), "map must be sorted");
};
template <> struct HostRuntimeLibrary<double, LibraryVersion::PgmathFast> {
  static constexpr HostRuntimeFunction table[]{
#define PGMATH_FAST
#define PGMATH_USE_D(name, func) REAL_FOLDER(name, func),
#include "flang/Evaluate/pgmath.h.inc"
  };
  static constexpr HostRuntimeMap map{table};
  static_assert(map.Verify(), "map must be sorted");
};
template <> struct HostRuntimeLibrary<float, LibraryVersion::PgmathRelaxed> {
  static constexpr HostRuntimeFunction table[]{
#define PGMATH_RELAXED
#define PGMATH_USE_S(name, func) REAL_FOLDER(name, func),
#include "flang/Evaluate/pgmath.h.inc"
  };
  static constexpr HostRuntimeMap map{table};
  static_assert(map.Verify(), "map must be sorted");
};
template <> struct HostRuntimeLibrary<double, LibraryVersion::PgmathRelaxed> {
  static constexpr HostRuntimeFunction table[]{
#define PGMATH_RELAXED
#define PGMATH_USE_D(name, func) REAL_FOLDER(name, func),
#include "flang/Evaluate/pgmath.h.inc"
  };
  static constexpr HostRuntimeMap map{table};
  static_assert(map.Verify(), "map must be sorted");
};
template <> struct HostRuntimeLibrary<float, LibraryVersion::PgmathPrecise> {
  static constexpr HostRuntimeFunction table[]{
#define PGMATH_PRECISE
#define PGMATH_USE_S(name, func) REAL_FOLDER(name, func),
#include "flang/Evaluate/pgmath.h.inc"
  };
  static constexpr HostRuntimeMap map{table};
  static_assert(map.Verify(), "map must be sorted");
};
template <> struct HostRuntimeLibrary<double, LibraryVersion::PgmathPrecise> {
  static constexpr HostRuntimeFunction table[]{
#define PGMATH_PRECISE
#define PGMATH_USE_D(name, func) REAL_FOLDER(name, func),
#include "flang/Evaluate/pgmath.h.inc"
  };
  static constexpr HostRuntimeMap map{table};
  static_assert(map.Verify(), "map must be sorted");
};

// TODO: double _Complex/float _Complex have been removed from llvm flang
// pgmath.h.inc because they caused warnings, they need to be added back
// so that the complex pgmath versions can be used when requested.

#endif /* LINK_WITH_LIBPGMATH */

// Helper to check if a HostRuntimeLibrary specialization exists
template <typename T, typename = void> struct IsAvailable : std::false_type {};
template <typename T>
struct IsAvailable<T, decltype((void)T::table, void())> : std::true_type {};
// Define helpers to find host runtime library map according to desired version
// and type.
template <typename HostT, LibraryVersion version>
static const HostRuntimeMap *GetHostRuntimeMapHelper(
    [[maybe_unused]] DynamicType resultType) {
  // A library must only be instantiated if LibraryVersion is
  // available on the host and if HostT maps to a Fortran type.
  // For instance, whenever long double and double are both 64-bits, double
  // is mapped to Fortran 64bits real type, and long double will be left
  // unmapped.
  if constexpr (host::FortranTypeExists<HostT>()) {
    using Lib = HostRuntimeLibrary<HostT, version>;
    if constexpr (IsAvailable<Lib>::value) {
      if (host::FortranType<HostT>{}.GetType() == resultType) {
        return &Lib::map;
      }
    }
  }
  return nullptr;
}
template <LibraryVersion version>
static const HostRuntimeMap *GetHostRuntimeMapVersion(DynamicType resultType) {
  if (resultType.category() == TypeCategory::Real) {
    if (const auto *map{GetHostRuntimeMapHelper<float, version>(resultType)}) {
      return map;
    }
    if (const auto *map{GetHostRuntimeMapHelper<double, version>(resultType)}) {
      return map;
    }
    if (const auto *map{
            GetHostRuntimeMapHelper<long double, version>(resultType)}) {
      return map;
    }
  }
  if (resultType.category() == TypeCategory::Complex) {
    if (const auto *map{GetHostRuntimeMapHelper<std::complex<float>, version>(
            resultType)}) {
      return map;
    }
    if (const auto *map{GetHostRuntimeMapHelper<std::complex<double>, version>(
            resultType)}) {
      return map;
    }
    if (const auto *map{
            GetHostRuntimeMapHelper<std::complex<long double>, version>(
                resultType)}) {
      return map;
    }
  }
  return nullptr;
}
static const HostRuntimeMap *GetHostRuntimeMap(
    LibraryVersion version, DynamicType resultType) {
  switch (version) {
  case LibraryVersion::Libm:
    return GetHostRuntimeMapVersion<LibraryVersion::Libm>(resultType);
  case LibraryVersion::LibmExtensions:
    return GetHostRuntimeMapVersion<LibraryVersion::LibmExtensions>(resultType);
  case LibraryVersion::PgmathPrecise:
    return GetHostRuntimeMapVersion<LibraryVersion::PgmathPrecise>(resultType);
  case LibraryVersion::PgmathRelaxed:
    return GetHostRuntimeMapVersion<LibraryVersion::PgmathRelaxed>(resultType);
  case LibraryVersion::PgmathFast:
    return GetHostRuntimeMapVersion<LibraryVersion::PgmathFast>(resultType);
  }
  return nullptr;
}

static const HostRuntimeFunction *SearchInHostRuntimeMap(
    const HostRuntimeMap &map, const std::string &name, DynamicType resultType,
    const std::vector<DynamicType> &argTypes) {
  auto sameNameRange{map.equal_range(name)};
  for (const auto *iter{sameNameRange.first}; iter != sameNameRange.second;
       ++iter) {
    if (iter->resultType == resultType && iter->argumentTypes == argTypes) {
      return &*iter;
    }
  }
  return nullptr;
}

// Search host runtime libraries for an exact type match.
static const HostRuntimeFunction *SearchHostRuntime(const std::string &name,
    DynamicType resultType, const std::vector<DynamicType> &argTypes) {
  // TODO: When command line options regarding targeted numerical library is
  // available, this needs to be revisited to take it into account. So far,
  // default to libpgmath if F18 is built with it.
#if LINK_WITH_LIBPGMATH
  if (const auto *map{
          GetHostRuntimeMap(LibraryVersion::PgmathPrecise, resultType)}) {
    if (const auto *hostFunction{
            SearchInHostRuntimeMap(*map, name, resultType, argTypes)}) {
      return hostFunction;
    }
  }
  // Default to libm if functions or types are not available in pgmath.
#endif
  if (const auto *map{GetHostRuntimeMap(LibraryVersion::Libm, resultType)}) {
    if (const auto *hostFunction{
            SearchInHostRuntimeMap(*map, name, resultType, argTypes)}) {
      return hostFunction;
    }
  }
  if (const auto *map{
          GetHostRuntimeMap(LibraryVersion::LibmExtensions, resultType)}) {
    if (const auto *hostFunction{
            SearchInHostRuntimeMap(*map, name, resultType, argTypes)}) {
      return hostFunction;
    }
  }
  return nullptr;
}

// Return a DynamicType that can hold all values of a given type.
// This is used to allow 16bit float to be folded with 32bits and
// x87 float to be folded with IEEE 128bits.
static DynamicType BiggerType(DynamicType type) {
  if (type.category() == TypeCategory::Real ||
      type.category() == TypeCategory::Complex) {
    // 16 bits floats to IEEE 32 bits float
    if (type.kind() == common::RealKindForPrecision(11) ||
        type.kind() == common::RealKindForPrecision(8)) {
      return {type.category(), common::RealKindForPrecision(24)};
    }
    // x87 float to IEEE 128 bits float
    if (type.kind() == common::RealKindForPrecision(64)) {
      return {type.category(), common::RealKindForPrecision(113)};
    }
  }
  return type;
}

std::optional<HostRuntimeWrapper> GetHostRuntimeWrapper(const std::string &name,
    DynamicType resultType, const std::vector<DynamicType> &argTypes) {
  if (const auto *hostFunction{SearchHostRuntime(name, resultType, argTypes)}) {
    return hostFunction->folder;
  }
  // If no exact match, search with "bigger" types and insert type
  // conversions around the folder.
  std::vector<evaluate::DynamicType> biggerArgTypes;
  evaluate::DynamicType biggerResultType{BiggerType(resultType)};
  for (auto type : argTypes) {
    biggerArgTypes.emplace_back(BiggerType(type));
  }
  if (const auto *hostFunction{
          SearchHostRuntime(name, biggerResultType, biggerArgTypes)}) {
    return [hostFunction, resultType](
               FoldingContext &context, std::vector<Expr<SomeType>> &&args) {
      auto nArgs{args.size()};
      for (size_t i{0}; i < nArgs; ++i) {
        args[i] = Fold(context,
            ConvertToType(hostFunction->argumentTypes[i], std::move(args[i]))
                .value());
      }
      return Fold(context,
          ConvertToType(
              resultType, hostFunction->folder(context, std::move(args)))
              .value());
    };
  }
  return std::nullopt;
}
} // namespace Fortran::evaluate
