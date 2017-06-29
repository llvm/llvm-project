// RUN: %clang_cc1 -triple x86_64-apple-macosx10.12.0 -fexceptions -faligned-alloc-unavailable -std=c++1z -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.12.0 -fexceptions -std=c++1z -verify -DNO_ERRORS %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.12.0 -fexceptions -faligned-allocation -faligned-alloc-unavailable -std=c++14 -verify %s

namespace std {
  typedef decltype(sizeof(0)) size_t;
  enum class align_val_t : std::size_t {};
  struct nothrow_t {};
  nothrow_t nothrow;
}

void *operator new(std::size_t __sz, const std::nothrow_t&) noexcept;
void *operator new[](std::size_t __sz, const std::nothrow_t&) noexcept;

void *operator new(std::size_t __sz, std::align_val_t, const std::nothrow_t&) noexcept;
void *operator new[](std::size_t __sz, std::align_val_t, const std::nothrow_t&) noexcept;
void operator delete(void *, std::align_val_t, const std::nothrow_t&);
void operator delete[](void *, std::align_val_t, const std::nothrow_t&);
void operator delete(void*, std::size_t, std::align_val_t) noexcept;
void operator delete[](void*, std::size_t, std::align_val_t) noexcept;

void *operator new(std::size_t, std::align_val_t, long long);

struct alignas(256) OveralignedS {
  int x[16];
};

struct S {
  int x[16];
};

void test() {
  auto *p = new S;
  delete p;
  p = new (std::nothrow) S;

  auto *pa = new S[4];
  delete[] pa;
  pa = new (std::nothrow) S[4];
}

void testOveraligned() {
  auto *p = new OveralignedS;
  p = new ((std::align_val_t)8) OveralignedS;
  delete p;
  p = new (std::nothrow) OveralignedS;

  auto *pa = new OveralignedS[4];
  pa = new ((std::align_val_t)8) OveralignedS[4];
  delete[] pa;
  pa = new (std::nothrow) OveralignedS[4];
  // No error here since it is not calling a replaceable allocation function.
  p = new ((std::align_val_t)8, 10LL) OveralignedS;
}

#ifdef NO_ERRORS
// expected-no-diagnostics
#else
// expected-error@-16 {{aligned allocation function of type 'void *(unsigned long, enum std::align_val_t)' possibly unavailable on}}
// expected-note@-17 {{if you supply your own aligned allocation functions}}
// expected-error@-18 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' possibly unavailable on}}
// expected-note@-19 {{if you supply your own aligned allocation functions}}

// expected-error@-20 {{aligned allocation function of type 'void *(unsigned long, enum std::align_val_t)' possibly unavailable on}}
// expected-note@-21 {{if you supply your own aligned allocation functions}}
// expected-error@-22 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' possibly unavailable on}}
// expected-note@-23 {{if you supply your own aligned allocation functions}}

// expected-error@-24 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' possibly unavailable on}}
// expected-note@-25 {{if you supply your own aligned allocation functions}}

// expected-error@-26 {{aligned allocation function of type 'void *(std::size_t, std::align_val_t, const std::nothrow_t &) noexcept' possibly unavailable on}}
// expected-note@-27 {{if you supply your own aligned allocation functions}}
// expected-error@-28 {{aligned deallocation function of type 'void (void *, std::align_val_t, const std::nothrow_t &) noexcept' possibly unavailable on}}
// expected-note@-29 {{if you supply your own aligned allocation functions}}

// expected-error@-29 {{aligned allocation function of type 'void *(unsigned long, enum std::align_val_t)' possibly unavailable on}}
// expected-note@-30 {{if you supply your own aligned allocation functions}}
// expected-error@-31 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' possibly unavailable on}}
// expected-note@-32 {{if you supply your own aligned allocation functions}}

// expected-error@-33 {{aligned allocation function of type 'void *(unsigned long, enum std::align_val_t)' possibly unavailable on}}
// expected-note@-34 {{if you supply your own aligned allocation functions}}
// expected-error@-35 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' possibly unavailable on}}
// expected-note@-36 {{if you supply your own aligned allocation functions}}

// expected-error@-37 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' possibly unavailable on}}
// expected-note@-38 {{if you supply your own aligned allocation functions}}

// expected-error@-39 {{aligned allocation function of type 'void *(std::size_t, std::align_val_t, const std::nothrow_t &) noexcept' possibly unavailable on}}
// expected-note@-40 {{if you supply your own aligned allocation functions}}
// expected-error@-41 {{aligned deallocation function of type 'void (void *, std::align_val_t, const std::nothrow_t &) noexcept' possibly unavailable on}}
// expected-note@-42 {{if you supply your own aligned allocation functions}}

#endif

// No errors if user-defined aligned allocation functions are available.
void *operator new(std::size_t __sz, std::align_val_t) {
  static char array[256];
  return &array;
}

void operator delete(void *p, std::align_val_t) {
}

void testOveraligned2() {
  auto p = new ((std::align_val_t)8) OveralignedS;
  delete p;
}
