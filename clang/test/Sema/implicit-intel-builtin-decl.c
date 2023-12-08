// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-feature +sse2 -fsyntax-only -verify=expected,c %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -target-feature +sse2 -fsyntax-only -verify=expected,cxx %s -x c++

void f(void) {
  (void)_mm_getcsr(); // cxx-warning{{implicitly declaring library function '_mm_getcsr'}} \
                         c-error{{call to undeclared library function '_mm_getcsr'}} \
                         expected-note{{include the header <xmmintrin.h> or explicitly provide a declaration for '_mm_getcsr'}}
  _mm_setcsr(1); // cxx-warning{{implicitly declaring library function '_mm_setcsr'}} \
                    c-error{{call to undeclared library function '_mm_setcsr'}} \
                    expected-note{{include the header <xmmintrin.h> or explicitly provide a declaration for '_mm_setcsr'}}
  _mm_sfence(); // cxx-warning{{implicitly declaring library function '_mm_sfence'}} \
                   c-error{{call to undeclared library function '_mm_sfence'}} \
                   expected-note{{include the header <xmmintrin.h> or explicitly provide a declaration for '_mm_sfence'}}

  _mm_clflush((void*)0); // cxx-warning{{implicitly declaring library function '_mm_clflush'}} \
                            c-error{{call to undeclared library function '_mm_clflush'}} \
                            expected-note{{include the header <emmintrin.h> or explicitly provide a declaration for '_mm_clflush'}}
  _mm_lfence(); // cxx-warning{{implicitly declaring library function '_mm_lfence'}} \
                   c-error{{call to undeclared library function '_mm_lfence'}} \
                   expected-note{{include the header <emmintrin.h> or explicitly provide a declaration for '_mm_lfence'}}
  _mm_mfence(); // cxx-warning{{implicitly declaring library function '_mm_mfence'}} \
                   c-error{{call to undeclared library function '_mm_mfence'}} \
                   expected-note{{include the header <emmintrin.h> or explicitly provide a declaration for '_mm_mfence'}}
  _mm_pause(); // cxx-warning{{implicitly declaring library function '_mm_pause'}} \
                  c-error{{call to undeclared library function '_mm_pause'}} \
                  expected-note{{include the header <emmintrin.h> or explicitly provide a declaration for '_mm_pause'}}
}

unsigned int _mm_getcsr(void);
void _mm_setcsr(unsigned int);
void _mm_sfence(void);

void _mm_clflush(void const *);
void _mm_lfence(void);
void _mm_mfence(void);
void _mm_pause(void);

void g(void) {
  (void)_mm_getcsr();
  _mm_setcsr(1);
  _mm_sfence();

  _mm_clflush((void*)0);
  _mm_lfence();
  _mm_mfence();
  _mm_pause();
}
