// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors -DNOEXCEPT="throw()" -DBAD_ALLOC="throw(std::bad_alloc)"
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors -DNOEXCEPT=noexcept -DBAD_ALLOC=
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors -DNOEXCEPT=noexcept -DBAD_ALLOC=
// RUN: %clang_cc1 -std=c++1z %s -verify -fexceptions -fcxx-exceptions -pedantic-errors -DNOEXCEPT=noexcept -DBAD_ALLOC=

// dr412: 3.4
// lwg404: yes
// lwg2340: yes

// FIXME: __SIZE_TYPE__ expands to 'long long' on some targets.
__extension__ typedef __SIZE_TYPE__ size_t;
namespace std { struct bad_alloc {}; }

inline void* operator new(size_t) BAD_ALLOC; // expected-error {{cannot be declared 'inline'}}
inline void* operator new[](size_t) BAD_ALLOC; // expected-error {{cannot be declared 'inline'}}
inline void operator delete(void*) NOEXCEPT; // expected-error {{cannot be declared 'inline'}}
inline void operator delete[](void*) NOEXCEPT; // expected-error {{cannot be declared 'inline'}}
#ifdef __cpp_sized_deallocation
inline void operator delete(void*, size_t) NOEXCEPT; // expected-error {{cannot be declared 'inline'}}
inline void operator delete[](void*, size_t) NOEXCEPT; // expected-error {{cannot be declared 'inline'}}
#endif
