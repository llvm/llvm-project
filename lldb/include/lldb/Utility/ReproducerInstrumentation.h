//===-- ReproducerInstrumentation.h -----------------------------*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_REPRODUCERINSTRUMENTATION_H
#define LLDB_UTILITY_REPRODUCERINSTRUMENTATION_H

#include "lldb/Utility/FileSpec.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/Logging.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <map>
#include <thread>
#include <type_traits>

template <typename T,
          typename std::enable_if<std::is_fundamental<T>::value, int>::type = 0>
inline void stringify_append(llvm::raw_string_ostream &ss, const T &t) {
  ss << t;
}

template <typename T, typename std::enable_if<!std::is_fundamental<T>::value,
                                              int>::type = 0>
inline void stringify_append(llvm::raw_string_ostream &ss, const T &t) {
  ss << &t;
}

template <typename T>
inline void stringify_append(llvm::raw_string_ostream &ss, T *t) {
  ss << reinterpret_cast<void *>(t);
}

template <typename T>
inline void stringify_append(llvm::raw_string_ostream &ss, const T *t) {
  ss << reinterpret_cast<const void *>(t);
}

template <>
inline void stringify_append<char>(llvm::raw_string_ostream &ss,
                                   const char *t) {
  ss << '\"' << t << '\"';
}

template <>
inline void stringify_append<std::nullptr_t>(llvm::raw_string_ostream &ss,
                                             const std::nullptr_t &t) {
  ss << "\"nullptr\"";
}

template <typename Head>
inline void stringify_helper(llvm::raw_string_ostream &ss, const Head &head) {
  stringify_append(ss, head);
}

template <typename Head, typename... Tail>
inline void stringify_helper(llvm::raw_string_ostream &ss, const Head &head,
                             const Tail &... tail) {
  stringify_append(ss, head);
  ss << ", ";
  stringify_helper(ss, tail...);
}

template <typename... Ts> inline std::string stringify_args(const Ts &... ts) {
  std::string buffer;
  llvm::raw_string_ostream ss(buffer);
  stringify_helper(ss, ts...);
  return ss.str();
}

#define LLDB_CONSTRUCT_(T, Class, ...)                                         \
  lldb_private::repro::Recorder _recorder(LLVM_PRETTY_FUNCTION);

#define LLDB_RECORD_CONSTRUCTOR(Class, Signature, ...)                         \
  LLDB_CONSTRUCT_(Class Signature, this, __VA_ARGS__)

#define LLDB_RECORD_CONSTRUCTOR_NO_ARGS(Class)                                 \
  LLDB_CONSTRUCT_(Class(), this, lldb_private::repro::EmptyArg())

#define LLDB_RECORD_(T1, T2, ...)                                              \
  lldb_private::repro::Recorder _recorder(LLVM_PRETTY_FUNCTION,                \
                                          stringify_args(__VA_ARGS__));

#define LLDB_RECORD_METHOD(Result, Class, Method, Signature, ...)              \
  LLDB_RECORD_(Result(Class::*) Signature, (&Class::Method), this, __VA_ARGS__)

#define LLDB_RECORD_METHOD_CONST(Result, Class, Method, Signature, ...)        \
  LLDB_RECORD_(Result(Class::*) Signature const, (&Class::Method), this,       \
               __VA_ARGS__)

#define LLDB_RECORD_METHOD_NO_ARGS(Result, Class, Method)                      \
  LLDB_RECORD_(Result (Class::*)(), (&Class::Method), this)

#define LLDB_RECORD_METHOD_CONST_NO_ARGS(Result, Class, Method)                \
  LLDB_RECORD_(Result (Class::*)() const, (&Class::Method), this)

#define LLDB_RECORD_STATIC_METHOD(Result, Class, Method, Signature, ...)       \
  LLDB_RECORD_(Result(*) Signature, (&Class::Method), __VA_ARGS__)

#define LLDB_RECORD_STATIC_METHOD_NO_ARGS(Result, Class, Method)               \
  LLDB_RECORD_(Result (*)(), (&Class::Method), lldb_private::repro::EmptyArg())

#define LLDB_RECORD_CHAR_PTR_(T1, T2, StrOut, ...)                             \
  lldb_private::repro::Recorder _recorder(LLVM_PRETTY_FUNCTION,                \
                                          stringify_args(__VA_ARGS__));

#define LLDB_RECORD_CHAR_PTR_METHOD(Result, Class, Method, Signature, StrOut,  \
                                    ...)                                       \
  LLDB_RECORD_CHAR_PTR_(Result(Class::*) Signature, (&Class::Method), StrOut,  \
                        this, __VA_ARGS__)

#define LLDB_RECORD_CHAR_PTR_METHOD_CONST(Result, Class, Method, Signature,    \
                                          StrOut, ...)                         \
  LLDB_RECORD_CHAR_PTR_(Result(Class::*) Signature const, (&Class::Method),    \
                        StrOut, this, __VA_ARGS__)

#define LLDB_RECORD_CHAR_PTR_STATIC_METHOD(Result, Class, Method, Signature,   \
                                           StrOut, ...)                        \
  LLDB_RECORD_CHAR_PTR_(Result(*) Signature, (&Class::Method), StrOut,         \
                        __VA_ARGS__)

#define LLDB_RECORD_RESULT(Result) Result;

/// The LLDB_RECORD_DUMMY macro is special because it doesn't actually record
/// anything. It's used to track API boundaries when we cannot record for
/// technical reasons.
#define LLDB_RECORD_DUMMY(Result, Class, Method, Signature, ...)               \
  lldb_private::repro::Recorder _recorder;

#define LLDB_RECORD_DUMMY_NO_ARGS(Result, Class, Method)                       \
  lldb_private::repro::Recorder _recorder;

namespace lldb_private {
namespace repro {

struct EmptyArg {};

/// RAII object that records function invocations and their return value.
///
/// API calls are only captured when the API boundary is crossed. Once we're in
/// the API layer, and another API function is called, it doesn't need to be
/// recorded.
///
/// When a call is recored, its result is always recorded as well, even if the
/// function returns a void. For functions that return by value, RecordResult
/// should be used. Otherwise a sentinel value (0) will be serialized.
///
/// Because of the functional overlap between logging and recording API calls,
/// this class is also used for logging.
class Recorder {
public:
  Recorder();
  Recorder(llvm::StringRef pretty_func, std::string &&pretty_args = {});
  ~Recorder();

private:
  void UpdateBoundary();

  /// Whether this function call was the one crossing the API boundary.
  bool m_local_boundary = false;
};

} // namespace repro
} // namespace lldb_private

#endif // LLDB_UTILITY_REPRODUCERINSTRUMENTATION_H
