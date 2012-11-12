// RUN: %clang_cc1 -std=c++11 -fexceptions -fcxx-exceptions -fsyntax-only -verify %s

// Exception specification compatibility.
// We test function pointers, because functions have an extra rule in p4.

// Same type is compatible
extern void (*r1)() throw(int);
extern void (*r1)() throw(int);

// Typedefs don't matter.
typedef int INT;
extern void (*r2)() throw(int);
extern void (*r2)() throw(INT);

// Order doesn't matter.
extern void (*r3)() throw(int, float);
extern void (*r3)() throw(float, int);

// MS throw-any spec and no spec at all are compatible
extern void (*r4)();
extern void (*r4)() throw(...);

// throw(X) and no spec are not compatible
extern void (*r5)() throw(int); // expected-note {{previous declaration}}
extern void (*r5)(); // expected-error {{exception specification in declaration does not match}}

// For functions, we accept this with a warning.
extern void f5() throw(int); // expected-note {{previous declaration}}
extern void f5(); // expected-warning {{missing exception specification}}

// Different types are not compatible.
extern void (*r7)() throw(int); // expected-note {{previous declaration}}
extern void (*r7)() throw(float); // expected-error {{exception specification in declaration does not match}}

// Top-level const doesn't matter.
extern void (*r8)() throw(int);
extern void (*r8)() throw(const int);

// Multiple appearances don't matter.
extern void (*r9)() throw(int, int);
extern void (*r9)() throw(int, int);


// noexcept is compatible with itself
extern void (*r10)() noexcept;
extern void (*r10)() noexcept;

// noexcept(true) is compatible with noexcept
extern void (*r11)() noexcept;
extern void (*r11)() noexcept(true);

// noexcept(false) isn't
extern void (*r12)() noexcept; // expected-note {{previous declaration}}
extern void (*r12)() noexcept(false); // expected-error {{does not match}}

// The form of the boolean expression doesn't matter.
extern void (*r13)() noexcept(1 < 2);
extern void (*r13)() noexcept(2 > 1);

// noexcept(false) is incompatible with noexcept(true)
extern void (*r14)() noexcept(true); // expected-note {{previous declaration}}
extern void (*r14)() noexcept(false); // expected-error {{does not match}}

// noexcept(false) is compatible with itself
extern void (*r15)() noexcept(false);
extern void (*r15)() noexcept(false);

// noexcept(false) is compatible with MS throw(...)
extern void (*r16)() noexcept(false);
extern void (*r16)() throw(...);

// noexcept(false) is *not* compatible with no spec
extern void (*r17)(); // expected-note {{previous declaration}}
extern void (*r17)() noexcept(false); // expected-error {{does not match}}

// except for functions
void f17();
void f17() noexcept(false);

// noexcept(false) is compatible with dynamic specs that throw unless
// CWG 1073 resolution is accepted. Clang implements it.
//extern void (*r18)() throw(int);
//extern void (*r18)() noexcept(false);

// noexcept(true) is compatible with dynamic specs that don't throw
extern void (*r19)() throw();
extern void (*r19)() noexcept(true);

// The other way round doesn't work.
extern void (*r20)() throw(); // expected-note {{previous declaration}}
extern void (*r20)() noexcept(false); // expected-error {{does not match}}

extern void (*r21)() throw(int); // expected-note {{previous declaration}}
extern void (*r21)() noexcept(true); // expected-error {{does not match}}


// As a very special workaround, we allow operator new to match no spec
// with a throw(bad_alloc) spec, because C++0x makes an incompatible change
// here.
extern "C++" { namespace std { class bad_alloc {}; } }
typedef decltype(sizeof(int)) mysize_t;
void* operator new(mysize_t) throw(std::bad_alloc);
void* operator new(mysize_t);
void* operator new[](mysize_t);
void* operator new[](mysize_t) throw(std::bad_alloc);

