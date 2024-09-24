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

template<unsigned Val>
void depth_too_high_templ() {
  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc loop collapse(Val)
  for(;;)
    for(;;);
}

void depth_too_high() {
  depth_too_high_templ<3>(); // expected-note{{in instantiation of function template specialization}}

  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc loop collapse(3)
  for(;;)
    for(;;);

  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc loop collapse(three())
  for(;;)
    for(;;);

  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc loop collapse(ConvertsThree{})
  for(;;)
    for(;;);
}

template<typename T, unsigned Three>
void not_single_loop_templ() {
  T Arr[5];
  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1 2{{active 'collapse' clause defined here}}
#pragma acc loop collapse(3)
  for(auto x : Arr) {
    for(auto y : Arr){
      do{}while(true); // expected-error{{do loop cannot appear in intervening code of a 'loop' with a 'collapse' clause}}
    }
  }

  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1 2{{active 'collapse' clause defined here}}
#pragma acc loop collapse(Three)
  for(;;) {
    for(;;){
      do{}while(true); // expected-error{{do loop cannot appear in intervening code of a 'loop' with a 'collapse' clause}}
    }
  }

#pragma acc loop collapse(Three)
  for(;;) {
    for(;;){
      for(;;){
        do{}while(true);
      }
    }
  }
  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1 2{{active 'collapse' clause defined here}}
#pragma acc loop collapse(Three)
  for(auto x : Arr) {
    for(auto y: Arr) {
      do{}while(true); // expected-error{{do loop cannot appear in intervening code of a 'loop' with a 'collapse' clause}}
    }
  }

#pragma acc loop collapse(Three)
  for(auto x : Arr) {
    for(auto y: Arr) {
      for(auto z: Arr) {
        do{}while(true);
      }
    }
  }
}

void not_single_loop() {
  not_single_loop_templ<int, 3>(); // expected-note{{in instantiation of function template}}

  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc loop collapse(3)
  for(;;) {
    for(;;){
      for(;;);
    }
    while(true); // expected-error{{while loop cannot appear in intervening code of a 'loop' with a 'collapse' clause}}
  }

  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc loop collapse(3)
  for(;;) {
    for(;;){
      for(;;);
    }
    do{}while(true); // expected-error{{do loop cannot appear in intervening code of a 'loop' with a 'collapse' clause}}
  }

  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1 2{{active 'collapse' clause defined here}}
#pragma acc loop collapse(3)
  for(;;) {
    for(;;){
      while(true); // expected-error{{while loop cannot appear in intervening code of a 'loop' with a 'collapse' clause}}
    }
  }
  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1 2{{active 'collapse' clause defined here}}
#pragma acc loop collapse(3)
  for(;;) {
    for(;;){
      do{}while(true); // expected-error{{do loop cannot appear in intervening code of a 'loop' with a 'collapse' clause}}
    }
  }

#pragma acc loop collapse(2)
  for(;;) {
    for(;;){
      do{}while(true);
    }
  }
#pragma acc loop collapse(2)
  for(;;) {
    for(;;){
      while(true);
    }
  }

  int Arr[5];
  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1 2{{active 'collapse' clause defined here}}
#pragma acc loop collapse(3)
  for(auto x : Arr) {
    for(auto y : Arr){
      do{}while(true); // expected-error{{do loop cannot appear in intervening code of a 'loop' with a 'collapse' clause}}
    }
  }

  // expected-note@+1 {{active 'collapse' clause defined here}}
#pragma acc loop collapse(3)
  for (;;) {
    for (;;) {
      for(;;);
    }
    // expected-error@+1{{more than one for-loop in a loop associated with OpenACC 'loop' construct with a 'collapse' clause}}
    for(;;);
  }

  // expected-note@+1 {{active 'collapse' clause defined here}}
#pragma acc loop collapse(3)
  for (;;) {
    for (;;) {
      for(;;);
    // expected-error@+1{{more than one for-loop in a loop associated with OpenACC 'loop' construct with a 'collapse' clause}}
      for(;;);
    }
  }

  for(;;);
#pragma acc loop collapse(3)
  for (;;) {
    for (;;) {
      for (;;);
    }
  }
}

template<unsigned Two, unsigned Three>
void no_other_directives() {
#pragma acc loop collapse(Two)
  for(;;) {
    for (;;) { // last loop associated with the top level.
    // expected-error@+1{{'collapse' clause specifies a loop count greater than the number of available loops}}
#pragma acc loop collapse(Three) // expected-note 2{{active 'collapse' clause defined here}}
      for(;;) {
        for(;;) {
    // expected-error@+1{{OpenACC 'serial' construct cannot appear in intervening code of a 'loop' with a 'collapse' clause}}
#pragma acc serial
          ;
        }
      }
    }
  }
#pragma acc loop collapse(Two)// expected-note{{active 'collapse' clause defined here}}
  for(;;) {
    for (;;) { // last loop associated with the top level.
#pragma acc loop collapse(Three)
      for(;;) {
        for(;;) {
          for(;;);
        }
      }
    }
    // expected-error@+1{{OpenACC 'serial' construct cannot appear in intervening code of a 'loop' with a 'collapse' clause}}
#pragma acc serial
          ;
  }
}

void no_other_directives() {
  no_other_directives<2,3>(); // expected-note{{in instantiation of function template specialization}}

  // Ok, not inside the intervening list
#pragma acc loop collapse(2)
  for(;;) {
    for(;;) {
#pragma acc data // expected-warning{{OpenACC construct 'data' not yet implemented}}
    }
  }
  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc loop collapse(2)
  for(;;) {
    // expected-error@+1{{OpenACC 'data' construct cannot appear in intervening code of a 'loop' with a 'collapse' clause}}
#pragma acc data // expected-warning{{OpenACC construct 'data' not yet implemented}}
    for(;;) {
    }
  }
}

