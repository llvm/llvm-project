// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -fms-extensions -verify %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fsyntax-only -fms-extensions -verify %s  -DMSVC

// Export const variable initialization.

#ifdef MSVC
// expected-no-diagnostics
#endif

#ifndef MSVC
// expected-warning@+2 {{__declspec attribute 'dllexport' is not supported}}
#endif
__declspec(dllexport) int const x = 3;

namespace {
namespace named {
#ifndef MSVC
// expected-warning@+2 {{__declspec attribute 'dllexport' is not supported}}
#endif
__declspec(dllexport) int const x = 3;
}
} // namespace

namespace named1 {
namespace {
namespace named {
#ifndef MSVC
// expected-warning@+2 {{__declspec attribute 'dllexport' is not supported}}
#endif
__declspec(dllexport) int const x = 3;
}
} // namespace
} // namespace named1
