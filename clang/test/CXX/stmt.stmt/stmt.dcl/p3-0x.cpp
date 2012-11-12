// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

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

void f();
void test_Y() {
  goto end; // expected-error{{goto into protected scope}}
  Y y; // expected-note{{jump bypasses variable with a non-trivial destructor}}
 end:
  f();
  goto inner; // expected-error{{goto into protected scope}}
  {
    Y y2; // expected-note{{jump bypasses variable with a non-trivial destructor}}
  inner:
    f();    
  }
  return;
}

struct Z {
  Z operator=(const Z&);
};

void test_Z() {
  goto end;
  Z z;
 end:
  return;
}
