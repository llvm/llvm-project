// Tests that the implicit declared allocation functions
// are attached to the global module fragment.
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/foo.cppm -fsyntax-only -verify
// RUN: %clang_cc1 -std=c++20 %t/foo2.cppm -fsyntax-only -verify

//--- foo.cppm
export module foo;
export void alloc_wrapper() {
  void *a = ::operator new(32);
  // [basic.stc.dynamic.general]Note2
  //   The implicit declarations do not introduce the names std, std::size_­t,
  //   std::align_­val_­t, ..., However, referring to std or std::size_­t or
  //   std::align_­val_­t is ill-formed unless a standard library declaration
  //   ([cstddef.syn], [new.syn], [std.modules]) of that name precedes
  //   ([basic.lookup.general]) the use of that name.
  void *b = ::operator new((std::size_t)32); // expected-error {{use of undeclared identifier 'std'}}
  void *c = ::operator new((std::size_t)32, // expected-error {{use of undeclared identifier 'std'}}
                           (std::align_val_t)64); // expected-error {{use of undeclared identifier 'std'}}

  ::operator delete(a);
  ::operator delete(b, (std::size_t)32); // expected-error {{use of undeclared identifier 'std'}}
  ::operator delete(c, (std::size_t)32,  // expected-error {{use of undeclared identifier 'std'}}
                       (std::align_val_t)64); // expected-error {{use of undeclared identifier 'std'}}
}

//--- new
namespace std {
  using size_t = decltype(sizeof(0));
  enum class align_val_t : size_t {};
}

[[nodiscard]] void *operator new(std::size_t);
[[nodiscard]] void *operator new(std::size_t, std::align_val_t);
[[nodiscard]] void *operator new[](std::size_t);
[[nodiscard]] void *operator new[](std::size_t, std::align_val_t);
void operator delete(void*) noexcept;
void operator delete(void*, std::size_t) noexcept;
void operator delete(void*, std::align_val_t) noexcept;
void operator delete(void*, std::size_t, std::align_val_t) noexcept;
void operator delete[](void*, std::size_t, std::align_val_t) noexcept;
void operator delete[](void*, std::size_t) noexcept;
void operator delete[](void*, std::align_val_t) noexcept;
void operator delete[](void*, std::size_t, std::align_val_t) noexcept;

//--- foo2.cppm
// expected-no-diagnostics
module;
#include "new"
export module foo2;
export void alloc_wrapper() {
  void *a = ::operator new(32);
  void *b = ::operator new((std::size_t)32);
  void *c = ::operator new((std::size_t)32,
                           (std::align_val_t)64);

  ::operator delete(a);
  ::operator delete(b, (std::size_t)32);
  ::operator delete(c, (std::size_t)32,
                       (std::align_val_t)64);
}
