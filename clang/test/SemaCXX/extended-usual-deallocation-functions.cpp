// RUN: %clang_cc1 -fexceptions -std=c++2a -fsized-deallocation -fno-aligned-allocation -verify %s
// RUN: %clang_cc1 -fexceptions -std=c++17 -fsized-deallocation -fno-aligned-allocation -verify %s
// RUN: %clang_cc1 -fexceptions -std=c++14 -fsized-deallocation -faligned-allocation -DHAS_ALIGN -verify %s
// RUN: %clang_cc1 -fexceptions -std=c++11 -fsized-deallocation -faligned-allocation -DHAS_ALIGN -verify %s

// Test that we handle aligned deallocation, sized deallocation, and destroying
// delete as usual deallocation functions even if they are used as extensions
// prior to C++17.

namespace std {
using size_t = decltype(sizeof(0));
enum class align_val_t : size_t;

struct destroying_delete_t {
  struct __construct { explicit __construct() = default; };
  explicit destroying_delete_t(__construct) {}
};

inline constexpr destroying_delete_t destroying_delete(destroying_delete_t::__construct());
}

// FIXME: Should destroying delete really be on in all dialects by default?
struct A {
  void operator delete(void*) = delete;
  void operator delete(A*, std::destroying_delete_t) = delete; // expected-note {{deleted}}
};
void ATest(A* a) { delete a; } // expected-error {{deleted}}

struct B {
  void operator delete(void*) = delete; // expected-note {{deleted}}
  void operator delete(void*, std::size_t) = delete;
};
void BTest(B *b) { delete b; }// expected-error {{deleted}}


struct alignas(128) C {
#ifndef HAS_ALIGN
  // expected-note@+2 {{deleted}}
#endif
  void operator delete(void*) = delete;
#ifdef HAS_ALIGN
  // expected-note@+2 {{deleted}}
#endif
  void operator delete(void*, std::align_val_t) = delete;
};
void CTest(C *c) { delete c; } // expected-error {{deleted}}

struct D {
  void operator delete(void*) = delete;
  void operator delete(D*, std::destroying_delete_t) = delete; // expected-note {{deleted}}
  void operator delete(D*, std::destroying_delete_t, std::size_t) = delete;
  void operator delete(D*, std::destroying_delete_t, std::align_val_t) = delete;
  void operator delete(D*, std::destroying_delete_t, std::size_t, std::align_val_t) = delete;
};
void DTest(D *d) { delete d; } // expected-error {{deleted}}

struct alignas(128) E {
  void operator delete(void*) = delete;
  void operator delete(E*, std::destroying_delete_t) = delete;
  void operator delete(E*, std::destroying_delete_t, std::size_t) = delete;
  void operator delete(E*, std::destroying_delete_t, std::align_val_t) = delete;
  void operator delete(E*, std::destroying_delete_t, std::size_t, std::align_val_t) = delete;
#ifdef HAS_ALIGN
  // expected-note@-3 {{deleted}}
#else
  // expected-note@-7 {{deleted}}
#endif
};
void ETest(E *e) { delete e; } // expected-error {{deleted}}
