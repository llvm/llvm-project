// RUN: %clang_cc1 -fsyntax-only -verify %s -Wno-error=non-pod-varargs
// RUN: %clang_cc1 -fsyntax-only -verify %s -Wno-error=non-pod-varargs -std=c++98
// RUN: %clang_cc1 -fsyntax-only -verify %s -Wno-error=non-pod-varargs -std=c++11

extern char version[];

@protocol P;

class C {
public:
  C(int);
};

@interface D 
- (void)g:(int)a, ...;
@end

void t1(D *d)
{
  C c(10);

  [d g:10, c]; 
#if __cplusplus <= 199711L // C++03 or earlier modes
  // expected-warning@-2{{cannot pass object of non-POD type 'C' through variadic method; call will abort at runtime}}
#else
  // expected-no-diagnostics@-4
#endif
  [d g:10, version];
}

void t2(D *d, id p)
{
  [d g:10, p];
}

void t3(D *d, id<P> p)
{
  [d g:10, p];
}
