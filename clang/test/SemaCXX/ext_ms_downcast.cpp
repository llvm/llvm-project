// RUN: %clang_cc1 -fsyntax-only -fms-compatibility -verify %s
// RUN: %clang_cc1 -fsyntax-only -DNO_MS_COMPATIBILITY -verify %s

// Minimal reproducer.
class A {};
class B : A {}; // expected-note 2 {{implicitly declared private here}}

B* foo(A* p) {
  return static_cast<B*>(p);
#ifdef NO_MS_COMPATIBILITY
  // expected-error@-2 {{cannot cast private base class 'A' to 'B'}}
#else
  // expected-warning@-4 {{casting from private base class 'A' to derived class 'B' is a Microsoft extension}}
#endif
}

A* bar(B* p) {
  return static_cast<A*>(p); // expected-error {{cannot cast 'B' to its private base class 'A'}}
}

// from atlframe.h
template <class T>
class CUpdateUI {
public:
  CUpdateUI() {
    T* pT = static_cast<T*>(this);
#ifdef NO_MS_COMPATIBILITY
    // expected-error@-2 {{cannot cast private base class}}
#else
    // expected-warning@-4 {{casting from private base class 'CUpdateUI<CMDIFrame>' to derived class 'CMDIFrame' is a Microsoft extension}}
#endif
  }
};

// from sample WTL/MDIDocVw (mainfrm.h
class CMDIFrame : CUpdateUI<CMDIFrame> {};
// expected-note@-1 {{implicitly declared private here}}
// expected-note@-2 {{in instantiation of member function}}

CMDIFrame wndMain;
