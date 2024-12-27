// RUN: %clang_cc1 -triple x86_64-apple-macosx10.12.0 -fexceptions -faligned-alloc-unavailable -std=c++1z -verify -DMACOS %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.12.0 -fexceptions -std=c++1z -verify -DNO_ERRORS %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.12.0 -fexceptions -faligned-allocation -faligned-alloc-unavailable -std=c++14 -verify -DMACOS %s
// RUN: %clang_cc1 -triple arm64-apple-ios10.0.0 -fexceptions -faligned-alloc-unavailable -std=c++1z -verify -DIOS %s
// RUN: %clang_cc1 -triple arm64-apple-ios10.0.0 -fexceptions -std=c++1z -verify -DNO_ERRORS %s
// RUN: %clang_cc1 -triple arm64-apple-tvos10.0.0 -fexceptions -faligned-alloc-unavailable -std=c++1z -verify -DTVOS %s
// RUN: %clang_cc1 -triple arm64-apple-tvos10.0.0 -fexceptions -std=c++1z -verify -DNO_ERRORS %s
// RUN: %clang_cc1 -triple armv7k-apple-watchos3.0.0 -fexceptions -faligned-alloc-unavailable -std=c++1z -verify -DWATCHOS %s
// RUN: %clang_cc1 -triple armv7k-apple-watchos3.0.0 -fexceptions -std=c++1z -verify -DNO_ERRORS %s
// RUN: %clang_cc1 -triple s390x-none-zos -fexceptions -faligned-alloc-unavailable -std=c++1z -verify -DZOS %s
// RUN: %clang_cc1 -triple s390x-none-zos -fexceptions -std=c++1z -verify -DNO_ERRORS %s
// RUN: %clang_cc1 -triple s390x-none-zos -fexceptions -faligned-allocation -faligned-alloc-unavailable -std=c++14 -verify -DZOS %s

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
// expected-error-re@-16 {{aligned allocation function of type 'void *(unsigned long, enum std::align_val_t)' is {{only|not}} available on}}
// expected-note@-17 {{if you supply your own aligned allocation functions}}
// expected-error-re@-18 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' is {{only|not}} available on}}
// expected-note@-19 {{if you supply your own aligned allocation functions}}

// expected-error-re@-20 {{aligned allocation function of type 'void *(unsigned long, enum std::align_val_t)' is {{only|not}} available on}}
// expected-note@-21 {{if you supply your own aligned allocation functions}}
// expected-error-re@-22 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' is {{only|not}} available on}}
// expected-note@-23 {{if you supply your own aligned allocation functions}}

// expected-error-re@-24 {{aligned deallocation function of type 'void (void *, std::size_t, std::align_val_t) noexcept' is {{only|not}} available on}}
// expected-note@-25 {{if you supply your own aligned allocation functions}}

// expected-error-re@-26 {{aligned allocation function of type 'void *(std::size_t, std::align_val_t, const std::nothrow_t &) noexcept' is {{only|not}} available on}}
// expected-note@-27 {{if you supply your own aligned allocation functions}}
// expected-error-re@-28 {{aligned deallocation function of type 'void (void *, std::align_val_t, const std::nothrow_t &) noexcept' is {{only|not}} available on}}
// expected-note@-29 {{if you supply your own aligned allocation functions}}

// expected-error-re@-29 {{aligned allocation function of type 'void *(unsigned long, enum std::align_val_t)' is {{only|not}} available on}}
// expected-note@-30 {{if you supply your own aligned allocation functions}}
// expected-error-re@-31 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' is {{only|not}} available on}}
// expected-note@-32 {{if you supply your own aligned allocation functions}}

// expected-error-re@-33 {{aligned allocation function of type 'void *(unsigned long, enum std::align_val_t)' is {{only|not}} available on}}
// expected-note@-34 {{if you supply your own aligned allocation functions}}
// expected-error-re@-35 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' is {{only|not}} available on}}
// expected-note@-36 {{if you supply your own aligned allocation functions}}

// expected-error-re@-37 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' is {{only|not}} available on}}
// expected-note@-38 {{if you supply your own aligned allocation functions}}

// expected-error-re@-39 {{aligned allocation function of type 'void *(std::size_t, std::align_val_t, const std::nothrow_t &) noexcept' is {{only|not}} available on}}
// expected-note@-40 {{if you supply your own aligned allocation functions}}
// expected-error-re@-41 {{aligned deallocation function of type 'void (void *, std::align_val_t, const std::nothrow_t &) noexcept' is {{only|not}} available on}}
// expected-note@-42 {{if you supply your own aligned allocation functions}}

#endif

void testOveralignedCheckOS() {
  auto *p = new OveralignedS;
}

#ifdef NO_ERRORS
// expected-no-diagnostics
#else
#if defined(IOS)
// expected-error@-7 {{aligned allocation function of type 'void *(unsigned long, enum std::align_val_t)' is only available on iOS 11 or newer}}
// expected-error@-8 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' is only available on iOS 11 or newer}}}
#elif defined(TVOS)
// expected-error@-10 {{aligned allocation function of type 'void *(unsigned long, enum std::align_val_t)' is only available on tvOS 11 or newer}}}
// expected-error@-11 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' is only available on tvOS 11 or newer}}}
#elif defined(WATCHOS)
// expected-error@-13 {{aligned allocation function of type 'void *(unsigned long, enum std::align_val_t)' is only available on watchOS 4 or newer}}}
// expected-error@-14 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' is only available on watchOS 4 or newer}}}
#elif defined(MACOS)
// expected-error@-16 {{aligned allocation function of type 'void *(unsigned long, enum std::align_val_t)' is only available on macOS 10.13 or newer}}}
// expected-error@-17 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' is only available on macOS 10.13 or newer}}}
#elif defined(ZOS)
// expected-error@-19 {{aligned allocation function of type 'void *(unsigned long, enum std::align_val_t)' is not available on z/OS}}}
// expected-error@-20 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' is not available on z/OS}}}
#endif

// expected-note@-23 2 {{if you supply your own aligned allocation functions}}
#endif

// Test that diagnostics are produced when an unavailable aligned deallocation
// function is called from a deleting destructor.
struct alignas(256) OveralignedS2 {
  int a[4];
  virtual ~OveralignedS2();
};

OveralignedS2::~OveralignedS2() {}

#ifdef NO_ERRORS
// expected-no-diagnostics
#else
#if defined(IOS)
// expected-error@-6 {{aligned deallocation function of type 'void (void *, std::size_t, std::align_val_t) noexcept' is only available on iOS 11 or newer}}}
// expected-note@-7 {{if you supply your own aligned allocation functions}}
#elif defined(TVOS)
// expected-error@-9 {{aligned deallocation function of type 'void (void *, std::size_t, std::align_val_t) noexcept' is only available on tvOS 11 or newer}}}
// expected-note@-10 {{if you supply your own aligned allocation functions}}
#elif defined(WATCHOS)
// expected-error@-12 {{aligned deallocation function of type 'void (void *, std::size_t, std::align_val_t) noexcept' is only available on watchOS 4 or newer}}}
// expected-note@-13 {{if you supply your own aligned allocation functions}}
#elif defined(MACOS)
// expected-error@-15 {{aligned deallocation function of type 'void (void *, std::size_t, std::align_val_t) noexcept' is only available on macOS 10.13 or newer}}}
// expected-note@-16 {{if you supply your own aligned allocation functions}}
#elif defined(ZOS)
// expected-error@-18 {{aligned deallocation function of type 'void (void *, std::size_t, std::align_val_t) noexcept' is not available on z/OS}}}
// expected-note@-19 {{if you supply your own aligned allocation functions}}
#endif
#endif

void testExplicitOperatorNewDelete() {
  void *p = operator new(128);
  operator delete(p);
  p = operator new[](128);
  operator delete[](p);
  p = __builtin_operator_new(128);
  __builtin_operator_delete(p);
}

void testExplicitOperatorNewDeleteOveraligned() {
  void *p = operator new(128, (std::align_val_t)64);
  operator delete(p, (std::align_val_t)64);
  p = operator new[](128, (std::align_val_t)64);
  operator delete[](p, (std::align_val_t)64);
  p = __builtin_operator_new(128, (std::align_val_t)64);
  __builtin_operator_delete(p, (std::align_val_t)64);
}

#ifdef NO_ERRORS
// expected-no-diagnostics
#else
// expected-error-re@-11 {{aligned allocation function of type 'void *(unsigned long, enum std::align_val_t)' is {{only|not}} available on}}
// expected-note@-12 {{if you supply your own aligned allocation functions}}

// expected-error-re@-13 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' is {{only|not}} available on}}
// expected-note@-14 {{if you supply your own aligned allocation functions}}

// expected-error-re@-15 {{aligned allocation function of type 'void *(unsigned long, enum std::align_val_t)' is {{only|not}} available on}}
// expected-note@-16 {{if you supply your own aligned allocation functions}}

// expected-error-re@-17 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' is {{only|not}} available on}}
// expected-note@-18 {{if you supply your own aligned allocation functions}}

// expected-error-re@-19 {{aligned allocation function of type 'void *(unsigned long, enum std::align_val_t)' is {{only|not}} available on}}
// expected-note@-20 {{if you supply your own aligned allocation functions}}

// expected-error-re@-21 {{aligned deallocation function of type 'void (void *, enum std::align_val_t) noexcept' is {{only|not}} available on}}
// expected-note@-22 {{if you supply your own aligned allocation functions}}
#endif

// No errors if user-defined aligned allocation functions are available.
void *operator new(std::size_t __sz, std::align_val_t) {
  static char array[256];
  return &array;
}

void operator delete(void *p, std::align_val_t) {
}

void operator delete(void *p, std::size_t __sz, std::align_val_t) {
}

void testOveraligned2() {
  auto p = new ((std::align_val_t)8) OveralignedS;
  delete p;
}
