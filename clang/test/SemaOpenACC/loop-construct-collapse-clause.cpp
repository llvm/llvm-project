// RUN: %clang_cc1 %s -fopenacc -verify


void only_for_loops() {
  // expected-error@+3{{OpenACC 'loop' construct can only be applied to a 'for' loop}}
  // expected-note@+1{{'loop' construct is here}}
#pragma acc loop collapse(1)
  while(true);

  // expected-error@+3{{OpenACC 'loop' construct can only be applied to a 'for' loop}}
  // expected-note@+1{{'loop' construct is here}}
#pragma acc loop collapse(1)
  do{}while(true);

}

void only_one_on_loop() {
  // expected-error@+2{{OpenACC 'collapse' clause cannot appear more than once on a 'loop' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc loop collapse(1) collapse(1)
  for(;;);
}

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

template <typename T, int Val>
void negative_constexpr_templ() {
  // expected-error@+3 2{{OpenACC 'collapse' clause loop count must be a positive integer value, evaluated to 0}}
  // expected-note@#NCETN1{{in instantiation of function template specialization 'negative_constexpr_templ<int, -1>'}}
  // expected-note@#NCET1{{in instantiation of function template specialization 'negative_constexpr_templ<int, 1>'}}
#pragma acc loop collapse(T{})
  for(;;)
    for(;;);

  // expected-error@+1{{OpenACC 'collapse' clause loop count must be a positive integer value, evaluated to -1}}
#pragma acc loop collapse(Val)
  for(;;)
    for(;;);
}

void negative_constexpr(int i) {
#pragma acc loop collapse(2)
  for(;;)
    for(;;);

#pragma acc loop collapse(1)
  for(;;)
    for(;;);

  // expected-error@+1{{OpenACC 'collapse' clause loop count must be a positive integer value, evaluated to 0}}
#pragma acc loop collapse(0)
  for(;;)
    for(;;);

  // expected-error@+1{{OpenACC 'collapse' clause loop count must be a positive integer value, evaluated to -1}}
#pragma acc loop collapse(-1)
  for(;;)
    for(;;);

#pragma acc loop collapse(one())
  for(;;)
    for(;;);

  // expected-error@+1{{OpenACC 'collapse' clause loop count must be a positive integer value, evaluated to 0}}
#pragma acc loop collapse(zero())
  for(;;)
    for(;;);

  // expected-error@+1{{OpenACC 'collapse' clause loop count must be a positive integer value, evaluated to -1}}
#pragma acc loop collapse(neg())
  for(;;)
    for(;;);

  // expected-error@+1{{OpenACC 'collapse' clause loop count must be a constant expression}}
#pragma acc loop collapse(NotConstexpr{})
  for(;;)
    for(;;);

  // expected-error@+1{{OpenACC 'collapse' clause loop count must be a positive integer value, evaluated to -1}}
#pragma acc loop collapse(ConvertsNegative{})
  for(;;)
    for(;;);

#pragma acc loop collapse(ConvertsOne{})
  for(;;)
    for(;;);

  negative_constexpr_templ<int, -1>(); // #NCETN1

  negative_constexpr_templ<int, 1>(); // #NCET1
}

