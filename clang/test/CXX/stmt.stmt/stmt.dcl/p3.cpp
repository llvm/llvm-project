// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// PR10034
struct X {};

void exx(X) {}

int test_ptr10034(int argc, char **argv)
{
 if (argc > 3)
   goto end;

 X x;
 X xs[16];
 exx(x);

 end:
   if (argc > 1) {
   for (int i = 0; i < argc; ++i)
   {

   }
   }
   return 0;
}

struct Y {
  ~Y();
};

void test_Y() {
  goto end; // expected-error{{cannot jump from this goto statement to its label}}
  Y y; // expected-note{{jump bypasses variable with a non-trivial destructor}}
 end:
  return;
}

struct Z {
  Z operator=(const Z&);
};

void test_Z() {
  goto end;
#if __cplusplus <= 199711L
  // expected-error@-2 {{cannot jump from this goto statement to its label}}
#endif

  Z z;
#if __cplusplus <= 199711L
  // expected-note@-2 {{jump bypasses initialization of non-POD variable}}
#endif

 end:
  return;
}
