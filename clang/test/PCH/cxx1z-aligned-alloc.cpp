// No PCH:
// RUN: %clang_cc1 -pedantic -fsized-deallocation -std=c++1z -include %s -verify %s
//
// With PCH:
// RUN: %clang_cc1 -pedantic -fsized-deallocation -std=c++1z -emit-pch %s -o %t
// RUN: %clang_cc1 -pedantic -fsized-deallocation -std=c++1z -include-pch %t -verify %s

// expected-no-diagnostics

#ifndef HEADER
#define HEADER

using size_t = decltype(sizeof(0));

// Call the overaligned form of 'operator new'.
struct alignas(256) Q { int n; };
void *f() { return new Q; }

// Extract the std::align_val_t type from the implicit declaration of operator delete.
template<typename AlignValT>
AlignValT extract(void (*)(void*, size_t, AlignValT));
using T = decltype(extract(&operator delete));

#else

// ok, calls aligned allocation via placement syntax
void *q = new (T{16}) Q;

namespace std {
  enum class align_val_t : size_t {};
}

using T = std::align_val_t; // ok, same type

#endif
