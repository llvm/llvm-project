// RUN: %clang_cc1 -fsyntax-only -verify %s -Wno-error=non-pod-varargs

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

  [d g:10, c]; // expected-warning{{cannot pass object of non-POD type 'C' through variadic method; call will abort at runtime}}
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
