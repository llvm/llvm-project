//===--- __clang_interpreter_runtime_printvalue.h ---*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines runtime functions used to print STL components in
// clang-repl. They are very heavy so we should only include it once and on
// demand.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_INTERPRETER_RUNTIME_PRINT_VALUE_H
#define LLVM_CLANG_INTERPRETER_RUNTIME_PRINT_VALUE_H

#if !defined(__CLANG_REPL__)
#error "This file should only be included by clang-repl!"
#endif

namespace caas {
namespace runtime {}
} // namespace caas
using namespace caas::runtime;

#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

// FIXME: We should include it somewhere instead of duplicating it...
#if __has_attribute(visibility) &&                                             \
    (!(defined(_WIN32) || defined(__CYGWIN__)) ||                              \
     (defined(__MINGW32__) && defined(__clang__)))
#if defined(LLVM_BUILD_LLVM_DYLIB) || defined(LLVM_BUILD_SHARED_LIBS)
#define __REPL_EXTERNAL_VISIBILITY __attribute__((visibility("default")))
#else
#define __REPL_EXTERNAL_VISIBILITY
#endif
#else
#if defined(_WIN32)
#define __REPL_EXTERNAL_VISIBILITY __declspec(dllexport)
#endif
#endif

// Fallback.
template <class T,
          typename std::enable_if<!std::is_pointer<T>::value>::type * = nullptr>
inline std::string PrintValueRuntime(const T &) {
  return "{not representable}";
}

// Forward declare the pre-compiled printing functions.
#ifndef __DECL_PRINT_VALUE_RUNTIME
#define __DECL_PRINT_VALUE_RUNTIME(type)                                       \
  __REPL_EXTERNAL_VISIBILITY std::string PrintValueRuntime(const type *__Ptr)
__DECL_PRINT_VALUE_RUNTIME(void);        //
__DECL_PRINT_VALUE_RUNTIME(void *);      //
__DECL_PRINT_VALUE_RUNTIME(char *const); //
__DECL_PRINT_VALUE_RUNTIME(char *);      //
__DECL_PRINT_VALUE_RUNTIME(bool);
__DECL_PRINT_VALUE_RUNTIME(char);
__DECL_PRINT_VALUE_RUNTIME(signed char);
__DECL_PRINT_VALUE_RUNTIME(short);
__DECL_PRINT_VALUE_RUNTIME(unsigned short);
__DECL_PRINT_VALUE_RUNTIME(int);
__DECL_PRINT_VALUE_RUNTIME(unsigned int);
__DECL_PRINT_VALUE_RUNTIME(long);
__DECL_PRINT_VALUE_RUNTIME(unsigned long);
__DECL_PRINT_VALUE_RUNTIME(long long);
__DECL_PRINT_VALUE_RUNTIME(unsigned long long);
__DECL_PRINT_VALUE_RUNTIME(float);
__DECL_PRINT_VALUE_RUNTIME(double);
__DECL_PRINT_VALUE_RUNTIME(long double);
#endif

namespace __repl_runtime_detail {

// Custom void_t implementation for C++11 compatibility
template <typename... Ts> struct __repl_void_impl {
  typedef void type;
};

// Helper to deduce the type of the expression 'std::begin(std::declval<T &>())'
template <typename T>
using __repl_begin_result = decltype(std::begin(std::declval<T &>()));

// Helper to deduce the type of the expression 'std::end(std::declval<T &>())'
template <typename T>
using __repl_end_result = decltype(std::end(std::declval<T &>()));

// Type trait to check if a type is iterable
template <typename T, typename = void>
struct __is_iterable : std::false_type {};

template <typename T>
struct __is_iterable<T, typename __repl_void_impl<__repl_begin_result<T>,
                                                  __repl_end_result<T>>::type>
    : std::true_type {};

// Type trait to check if a type is std::pair
template <typename T> struct __is_pair : std::false_type {};

template <typename T, typename U>
struct __is_pair<std::pair<T, U>> : std::true_type {};

// Type trait to check if a type is std::map (or any associative container with
// mapped_type)
template <typename T, typename = void> struct __is_map : std::false_type {};

template <typename T>
struct __is_map<T, typename __repl_void_impl<typename T::mapped_type>::type>
    : std::true_type {};

// The type of the elements is std::pair, and the container is a map like type.
template <
    typename Container, typename Elt,
    typename std::enable_if<__is_pair<Elt>::value && __is_map<Container>::value,
                            bool>::type = true>
std::string __PrintCollectionElt(const Elt &Val) {
  return PrintValueRuntime(&Val.first) + " => " +
         PrintValueRuntime(&Val.second);
}

// The type of the elements is std::pair, and the container isn't a map-like
// type.
template <typename Container, typename Elt,
          typename std::enable_if<__is_pair<Elt>::value &&
                                      !__is_map<Container>::value,
                                  bool>::type = true>
std::string __PrintCollectionElt(const Elt &Val) {
  return TuplePairPrintValue(&Val);
}

template <typename Container, typename Elt,
          typename std::enable_if<!__is_pair<Elt>::value, bool>::type = true>
std::string __PrintCollectionElt(const Elt &Val) {
  return PrintValueRuntime(&Val);
}

template <class Tuple, std::size_t N = std::tuple_size<Tuple>(),
          std::size_t TupleSize = std::tuple_size<Tuple>()>
struct __TupleLikePrinter {
  static std::string print(const Tuple *T) {
    constexpr std::size_t EltNum = TupleSize - N;
    std::string Str;
    // Not the first element.
    if (EltNum != 0)
      Str += ", ";
    Str += PrintValueRuntime(&std::get<EltNum>(*T));
    // If N+1 is not smaller than the size of the tuple,
    // reroute the call to the printing function to the
    // no-op specialisation to stop recursion.
    constexpr std::size_t Nm1 = N - 1;
    Str += __TupleLikePrinter<Tuple, Nm1>::print((const Tuple *)T);
    return Str;
  }
};

template <class Tuple, std::size_t TupleSize>
struct __TupleLikePrinter<Tuple, 0, TupleSize> {
  static std::string print(const Tuple *T) { return ""; }
};

template <class T> inline std::string TuplePairPrintValue(const T *Val) {
  std::string Str("{ ");
  Str += __TupleLikePrinter<T>::print(Val);
  Str += " }";
  return Str;
}

struct __StdVectorBool {
  bool Value;
  __StdVectorBool(bool V) : Value(V) {}
};
template <typename T>
std::string __PrintCollectionElt(const __StdVectorBool &Val) {
  return PrintValueRuntime(&Val.Value);
}

} // namespace __repl_runtime_detail

template <typename Container,
          typename std::enable_if<
              __repl_runtime_detail::__is_iterable<Container>::value,
              bool>::type = true>
inline std::string PrintValueRuntime(const Container *C) {
  std::string Str("{ ");

  for (auto Beg = C->begin(), End = C->end(); Beg != End; Beg++) {
    if (Beg != C->begin())
      Str += ", ";
    Str += __repl_runtime_detail::__PrintCollectionElt<Container>(*Beg);
  }
  Str += " }";
  return Str;
}

template <typename T, size_t N>
inline std::string PrintValueRuntime(const T (*Obj)[N]) {
  if (N == 0)
    return "{}";

  std::string Str = "{ ";
  for (size_t Idx = 0; Idx < N; ++Idx) {
    Str += PrintValueRuntime(*Obj + Idx);
    if (Idx < N - 1)
      Str += ", ";
  }
  return Str + " }";
}

template <size_t N> inline std::string PrintValueRuntime(const char (*Obj)[N]) {
  const auto *Str = reinterpret_cast<const char *const>(Obj);
  return PrintValueRuntime(&Str);
}

// std::vector<bool>
inline std::string PrintValueRuntime(const std::vector<bool> *Val) {
  // Try our best to fix std::vector<bool> without too much of templated code.
  std::vector<__repl_runtime_detail::__StdVectorBool> ValFixed(Val->begin(),
                                                               Val->end());
  return PrintValueRuntime<decltype(ValFixed)>(&ValFixed);
}

// tuple
template <typename... Ts>
inline std::string PrintValueRuntime(const std::tuple<Ts...> *Val) {
  using T = std::tuple<Ts...>;
  return __repl_runtime_detail::TuplePairPrintValue<T>(Val);
}

// pair
template <typename... Ts>
inline std::string PrintValueRuntime(const std::pair<Ts...> *Val) {
  using T = std::pair<Ts...>;
  return __repl_runtime_detail::TuplePairPrintValue<T>(Val);
}

// unique_ptr
template <class T>
inline std::string PrintValueRuntime(const std::unique_ptr<T> *Val) {
  auto Ptr = Val->get();
  return "std::unique_ptr -> " + PrintValueRuntime((const void **)&Ptr);
}

// shared_ptr
template <class T>
inline std::string PrintValueRuntime(const std::shared_ptr<T> *Val) {
  auto Ptr = Val->get();
  return "std::shared_ptr -> " + PrintValueRuntime((const void **)&Ptr);
}

// weak_ptr
template <class T>
inline std::string PrintValueRuntime(const std::weak_ptr<T> *Val) {
  auto Ptr = Val->lock().get();
  return "std::weak_ptr -> " + PrintValueRuntime((const void **)&Ptr);
}

// string
template <class T>
inline std::string PrintValueRuntime(const std::basic_string<T> *Val) {
  const char *Chars = Val->c_str();
  return PrintValueRuntime((const char **)&Chars);
}
#endif
