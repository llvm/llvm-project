#ifndef BENCHMARK_INTERNAL_MACROS_H_
#define BENCHMARK_INTERNAL_MACROS_H_

#include "benchmark/macros.h"

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if defined(__clang__)
#define COMPILER_CLANG
#elif defined(_MSC_VER)
#define COMPILER_MSVC
#elif defined(__GNUC__)
#define COMPILER_GCC
#endif

#if __has_feature(cxx_attributes)
#define BENCHMARK_NORETURN [[noreturn]]
#elif defined(__GNUC__)
#define BENCHMARK_NORETURN __attribute__((noreturn))
#elif defined(COMPILER_MSVC)
#define BENCHMARK_NORETURN __declspec(noreturn)
#else
#define BENCHMARK_NORETURN
#endif

#if defined(__CYGWIN__)
#define BENCHMARK_OS_CYGWIN 1
#elif defined(_WIN32)
#define BENCHMARK_OS_WINDOWS 1
#elif defined(__APPLE__)
// TODO(ericwf) This doesn't actually check that it is a Mac OSX system. Just
// that it is an apple system.
#define BENCHMARK_OS_MACOSX 1
#elif defined(__FreeBSD__)
#define BENCHMARK_OS_FREEBSD 1
#elif defined(__linux__)
#define BENCHMARK_OS_LINUX 1
#endif

#endif  // BENCHMARK_INTERNAL_MACROS_H_
