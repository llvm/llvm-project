// RUN: %clang_cc1 -fsyntax-only -verify -fblocks %s

// PR9463
double *end;
void f() {
  {
    int end = 0;
    goto end;
    end = 1;
  }

 end:
  return;
}

void g() {
  end = 1; // expected-error{{assigning to 'double *' from incompatible type 'int'}}
}

void h(int end) {
  {
    goto end; // expected-error{{use of undeclared label 'end'}}
  }
}

void h2(int end) {
  {
    __label__ end;
    goto end;

  end:
    ::end = 0;
  }
 end:
  end = 1;
}

class X {
public:
  X();
};

void rdar9135994()
{
X:  
    goto X;
}

namespace PR9495 {
  struct NonPOD { NonPOD(); ~NonPOD(); };  
  
  void f(bool b) {
    NonPOD np;
    if (b) {
      goto undeclared; // expected-error{{use of undeclared label 'undeclared'}}
    }
  }

  void g() {
    (void)^(bool b){
      NonPOD np;
      if (b) {
        goto undeclared; // expected-error{{use of undeclared label 'undeclared'}}
      }
    };
  }
}


