// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 -std=c++11 -o - %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 -std=c++11 -o - %s -Wuninitialized

void foo() {
}

void xxx(int argc) {
  int x; // expected-note {{initialize the variable 'x' to silence this warning}}
#pragma omp parallel
  argc = x; // expected-warning {{variable 'x' is uninitialized when used here}}
}

#pragma omp parallel // expected-error {{unexpected OpenMP directive '#pragma omp parallel'}}

int a;
struct S;
S& bar();
int main(int argc, char **argv) {
  S &s = bar();
  #pragma omp parallel
  (void)&s;
  #pragma omp parallel { // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  foo();
  #pragma omp parallel ( // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  foo();
  #pragma omp parallel [ // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  foo();
  #pragma omp parallel ] // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  foo();
  #pragma omp parallel ) // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  foo();
  #pragma omp parallel } // expected-warning {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  foo();
  #pragma omp parallel
  // expected-warning@+1 {{extra tokens at the end of '#pragma omp parallel' are ignored}}
  #pragma omp parallel unknown()
  foo();
  L1:
    foo();
  #pragma omp parallel ordered // expected-error {{unexpected OpenMP clause 'ordered' in directive '#pragma omp parallel'}}
    ;
  #pragma omp parallel
  ;
  #pragma omp parallel
  {
    goto L1; // expected-error {{use of undeclared label 'L1'}}
    argc++;
  }

  for (int i = 0; i < 10; ++i) {
    switch(argc) {
     case (0):
      #pragma omp parallel
      {
        foo();
        break; // expected-error {{'break' statement not in loop or switch statement}}
        continue; // expected-error {{'continue' statement not in loop statement}}
      }
      default:
       break;
    }
  }
#pragma omp parallel default(none) // expected-note 2 {{explicit data sharing attribute requested here}}
  {
    ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}
    ++a;    // expected-error {{variable 'a' must have explicitly specified data sharing attributes}}
  }

  goto L2; // expected-error {{use of undeclared label 'L2'}}
  #pragma omp parallel
  L2:
  foo();
  #pragma omp parallel
  {
    return 1; // expected-error {{cannot return from OpenMP region}}
  }

  [[]] // expected-error {{an attribute list cannot appear here}}
  #pragma omp parallel
  for (int n = 0; n < 100; ++n) {}

  return 0;
}

struct a {
  static constexpr int b = 0;
};
template <bool> struct c;
template <typename d, typename e> bool operator<(d, e);
struct f {
  int cbegin;
};
class g {
  f blocks;
  void j();
};
template <typename> struct is_error_code_enum : a {};
struct h {
  template <typename i, typename = c<is_error_code_enum<i>::b>> h(i);
};
h operator<(h, h);
void g::j() {
#pragma omp parallel for default(none) if(a::b)
  for (auto a = blocks.cbegin; a < blocks; ++a) // expected-error 2 {{invalid operands to binary expression ('f' and 'int')}}
    ;
}
