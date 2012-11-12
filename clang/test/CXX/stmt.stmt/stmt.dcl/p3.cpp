// RUN: %clang_cc1 -fsyntax-only -verify %s

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
  goto end; // expected-error{{goto into protected scope}}
  Y y; // expected-note{{jump bypasses variable with a non-trivial destructor}}
 end:
  return;
}

struct Z {
  Z operator=(const Z&);
};

void test_Z() {
  goto end; // expected-error{{goto into protected scope}}
  Z z; // expected-note{{jump bypasses initialization of non-POD variable}}
 end:
  return;
}
