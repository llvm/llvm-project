// RUN: %clang_cc1 %s -fopenacc -verify

constexpr int three() { return 3; }
constexpr int one() { return 1; }
constexpr int neg() { return -1; }
constexpr int zero() { return 0; }

struct NotConstexpr {
  constexpr NotConstexpr(){};

  operator int(){ return 1; }
};
struct ConvertsNegative {
  constexpr ConvertsNegative(){};

  constexpr operator int(){ return -1; }
};
struct ConvertsOne{
  constexpr ConvertsOne(){};

  constexpr operator int(){ return 1; }
};

struct ConvertsThree{
  constexpr ConvertsThree(){};

  constexpr operator int(){ return 3; }
};

template<typename T, int Val>
void negative_zero_constexpr_templ() {
  // expected-error@+1 2{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to 0}}
#pragma acc loop tile(*, T{})
  for(;;)
    for(;;);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to -1}}
#pragma acc loop tile(Val, *)
  for(;;)
    for(;;);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to 0}}
#pragma acc loop tile(zero(), *)
  for(;;)
    for(;;);
}

void negative_zero_constexpr() {
  negative_zero_constexpr_templ<int, 1>(); // expected-note{{in instantiation of function template specialization}}
  negative_zero_constexpr_templ<int, -1>(); // expected-note{{in instantiation of function template specialization}}

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to 0}}
#pragma acc loop tile(0, *)
  for(;;)
    for(;;);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to 0}}
#pragma acc loop tile(1, 0)
  for(;;)
    for(;;);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to -1}}
#pragma acc loop tile(1, -1)
  for(;;)
    for(;;);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to -1}}
#pragma acc loop tile(-1, 0)
  for(;;)
    for(;;);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to 0}}
#pragma acc loop tile(zero(), 0)
  for(;;)
    for(;;);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to -1}}
#pragma acc loop tile(1, neg())
  for(;;)
    for(;;);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be an asterisk or a constant expression}}
#pragma acc loop tile(NotConstexpr{})
  for(;;);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to -1}}
#pragma acc loop tile(1, ConvertsNegative{})
  for(;;)
    for(;;);

#pragma acc loop tile(*, ConvertsOne{})
  for(;;)
    for(;;);

}
