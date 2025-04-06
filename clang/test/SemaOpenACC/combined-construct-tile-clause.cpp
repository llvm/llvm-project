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
#pragma acc serial loop tile(*, T{})
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to -1}}
#pragma acc parallel loop tile(Val, *)
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to 0}}
#pragma acc kernels loop tile(zero(), *)
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j);
}

void negative_zero_constexpr() {
  negative_zero_constexpr_templ<int, 1>(); // expected-note{{in instantiation of function template specialization}}
  negative_zero_constexpr_templ<int, -1>(); // expected-note{{in instantiation of function template specialization}}

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to 0}}
#pragma acc serial loop tile(0, *)
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to 0}}
#pragma acc parallel loop tile(1, 0)
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to -1}}
#pragma acc kernels loop tile(1, -1)
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to -1}}
#pragma acc parallel loop tile(-1, 0)
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to 0}}
#pragma acc serial loop tile(zero(), 0)
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to -1}}
#pragma acc kernels loop tile(1, neg())
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be an asterisk or a constant expression}}
#pragma acc parallel loop tile(NotConstexpr{})
  for(int i = 0; i < 5; ++i);

  // expected-error@+1{{OpenACC 'tile' clause size expression must be positive integer value, evaluated to -1}}
#pragma acc serial loop tile(1, ConvertsNegative{})
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j);

#pragma acc kernels loop tile(*, ConvertsOne{})
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j);
}

template<unsigned One>
void only_for_loops_templ() {
  // expected-note@+1{{'parallel loop' construct is here}}
#pragma acc parallel loop tile(One)
  // expected-error@+1{{OpenACC 'parallel loop' construct can only be applied to a 'for' loop}}
  while(true);

  // expected-note@+1{{'serial loop' construct is here}}
#pragma acc serial loop tile(One)
  // expected-error@+1{{OpenACC 'serial loop' construct can only be applied to a 'for' loop}}
  do {} while(true);

  // expected-error@+1{{'tile' clause specifies a loop count greater than the number of available loops}}
#pragma acc kernels loop tile(One, 2) // expected-note 2{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i)
      // expected-error@+1{{while loop cannot appear in intervening code of a 'kernels loop' with a 'tile' clause}}
    while(true);

  // expected-error@+1{{'tile' clause specifies a loop count greater than the number of available loops}}
#pragma acc serial loop tile(One, 2) // expected-note 2{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i)
      // expected-error@+1{{do loop cannot appear in intervening code of a 'serial loop' with a 'tile' clause}}
    do{}while(true);
}


void only_for_loops() {
  // expected-note@+1{{'parallel loop' construct is here}}
#pragma acc parallel loop tile(1)
  // expected-error@+1{{OpenACC 'parallel loop' construct can only be applied to a 'for' loop}}
  while(true);

  // expected-note@+1{{'serial loop' construct is here}}
#pragma acc serial loop tile(1)
  // expected-error@+1{{OpenACC 'serial loop' construct can only be applied to a 'for' loop}}
  do {} while(true);

  // expected-error@+1{{'tile' clause specifies a loop count greater than the number of available loops}}
#pragma acc kernels loop tile(1, 2) // expected-note 2{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i)
      // expected-error@+1{{while loop cannot appear in intervening code of a 'kernels loop' with a 'tile' clause}}
    while(true);

  // expected-error@+1{{'tile' clause specifies a loop count greater than the number of available loops}}
#pragma acc parallel loop tile(1, 2) // expected-note 2{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i)
      // expected-error@+1{{do loop cannot appear in intervening code of a 'parallel loop' with a 'tile' clause}}
    do{}while(true);
}

void only_one_on_loop() {
  // expected-error@+2{{OpenACC 'tile' clause cannot appear more than once on a 'serial loop' directive}}
  // expected-note@+1{{previous clause is here}}
#pragma acc serial loop tile(1) tile(1)
  for(int i = 0; i < 5; ++i);
}

template<unsigned Val>
void depth_too_high_templ() {
  // expected-error@+1{{'tile' clause specifies a loop count greater than the number of available loops}}
#pragma acc kernels loop tile (Val, *, Val) // expected-note{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j);

  // expected-error@+1{{'tile' clause specifies a loop count greater than the number of available loops}}
#pragma acc parallel loop tile (Val, *, Val) // expected-note 2{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j)
      // expected-error@+1{{while loop cannot appear in intervening code of a 'parallel loop' with a 'tile' clause}}
      while(true);

  // expected-error@+1{{'tile' clause specifies a loop count greater than the number of available loops}}
#pragma acc serial loop tile (Val, *, Val) // expected-note 2{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j)
      // expected-error@+1{{do loop cannot appear in intervening code of a 'serial loop' with a 'tile' clause}}
      do{}while(true);

  int Arr[Val+5];

  // expected-error@+1{{'tile' clause specifies a loop count greater than the number of available loops}}
#pragma acc kernels loop tile (Val, *, Val) // expected-note 2{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i)
    for(auto x : Arr)
      // expected-error@+1{{while loop cannot appear in intervening code of a 'kernels loop' with a 'tile' clause}}
      while(true)
        for(int j = 0; j < 5; ++j);

#pragma acc parallel loop tile (Val, *, Val)
  for(int i = 0; i < 5; ++i)
    for(auto x : Arr)
      for(int j = 0; j < 5; ++j)
        while(true);
}

void depth_too_high() {
  depth_too_high_templ<3>();

int Arr[5];

  // expected-error@+1{{'tile' clause specifies a loop count greater than the number of available loops}}
#pragma acc serial loop tile (1, *, 3) // expected-note{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j);

  // expected-error@+1{{'tile' clause specifies a loop count greater than the number of available loops}}
#pragma acc parallel loop tile (1, *, 3) // expected-note 2{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j)
      // expected-error@+1{{while loop cannot appear in intervening code of a 'parallel loop' with a 'tile' clause}}
      while(true);

  // expected-error@+1{{'tile' clause specifies a loop count greater than the number of available loops}}
#pragma acc parallel loop tile (1, *, 3) // expected-note 2{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j)
      // expected-error@+1{{do loop cannot appear in intervening code of a 'parallel loop' with a 'tile' clause}}
      do{}while(true);

  // expected-error@+1{{'tile' clause specifies a loop count greater than the number of available loops}}
#pragma acc parallel loop tile (1, *, 3) // expected-note 2{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i)
    for(int j = 0; j < 5; ++j)
      // expected-error@+1{{while loop cannot appear in intervening code of a 'parallel loop' with a 'tile' clause}}
      while(true)
        for(int j = 0; j < 5; ++j);

#pragma acc parallel loop tile (1, *, 3)
  for(int i = 0; i < 5; ++i)
    for(auto x : Arr)
      for(int j = 0; j < 5; ++j)
        while(true);
}

template<unsigned Val>
void not_single_loop_templ() {

  int Arr[Val];

#pragma acc parallel loop tile (Val, *, 3) // expected-note{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i) {
    for (auto x : Arr)
      for(int k = 0; k < 5; ++k);
  // expected-error@+1{{more than one for-loop in a loop associated with OpenACC 'parallel loop' construct with a 'tile' clause}}
    for(int j = 0; j < 5; ++j)
      for(int k = 0; k < 5; ++k);
  }
}

void not_single_loop() {
  not_single_loop_templ<3>(); // no diagnostic, was diagnosed in phase 1.

  int Arr[5];

#pragma acc parallel loop tile (1, *, 3)// expected-note{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i) {
    for (auto x : Arr)
      for(int k = 0; k < 5; ++k);
  // expected-error@+1{{more than one for-loop in a loop associated with OpenACC 'parallel loop' construct with a 'tile' clause}}
    for(int j = 0; j < 5; ++j)
      for(int k = 0; k < 5; ++k);
  }
}

template<unsigned Val>
void no_other_directives_templ() {

  int Arr[Val];

#pragma acc parallel loop tile (Val, *, 3) // expected-note{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i) {
    for (auto x : Arr) {
  // expected-error@+1{{OpenACC 'serial' construct cannot appear in intervening code of a 'parallel loop' with a 'tile' clause}}
#pragma acc serial
      ;
      for(int j = 0; j < 5; ++j);
    }
  }

  // OK, in innermost
#pragma acc parallel loop tile (Val, *, 3)
  for(int i = 0; i < 5; ++i) {
    for(int j = 0; j < 5; ++j) {
      for (auto x : Arr) {
#pragma acc serial
      ;
      }
    }
  }
}

void no_other_directives() {
  no_other_directives_templ<3>();
  int Arr[5];

#pragma acc parallel loop tile (1, *, 3) // expected-note{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i) {
    for (auto x : Arr) {
  // expected-error@+1{{OpenACC 'serial' construct cannot appear in intervening code of a 'parallel loop' with a 'tile' clause}}
#pragma acc serial
      ;
      for(int j = 0; j < 5; ++j);
    }
  }

  // OK, in innermost
#pragma acc parallel loop tile (3, *, 3)
  for(int i = 0; i < 5; ++i) {
    for(int j = 0; j < 5; ++j) {
      for (auto x : Arr) {
#pragma acc serial
      ;
      }
    }
  }
}

void call();
template<unsigned Val>
void intervening_templ() {
#pragma acc parallel loop tile(1, Val, *) // expected-note{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i) {
    //expected-error@+1{{inner loops must be tightly nested inside a 'tile' clause on a 'parallel loop' construct}}
    call();
    for(int j = 0; j < 5; ++j)
      for(int k = 0; k < 5; ++k);
  }

#pragma acc parallel loop tile(1, Val, *) // expected-note{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i) {
    //expected-error@+1{{inner loops must be tightly nested inside a 'tile' clause on a 'parallel loop' construct}}
    unsigned I;
    for(int j = 0; j < 5; ++j)
      for(int k = 0; k < 5; ++k);
  }

#pragma acc parallel loop tile(1, Val, *)
  // expected-error@+2{{OpenACC 'parallel loop' construct must have a terminating condition}}
  // expected-note@-2{{'parallel loop' construct is here}}
  for(int i = 0;;++i) {
  // expected-error@+2{{OpenACC 'parallel loop' construct must have a terminating condition}}
  // expected-note@-5{{'parallel loop' construct is here}}
    for(int j = 0;;++j)
  // expected-error@+2{{OpenACC 'parallel loop' construct must have a terminating condition}}
  // expected-note@-8{{'parallel loop' construct is here}}
      for(int k = 0;;++k)
        call();
  }
}

void intervening() {
  intervening_templ<3>();

#pragma acc parallel loop tile(1, 2, *) // expected-note{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i) {
    //expected-error@+1{{inner loops must be tightly nested inside a 'tile' clause on a 'parallel loop' construct}}
    call();
    for(int j = 0; j < 5; ++j)
      for(int k = 0; k < 5; ++k);
  }

#pragma acc parallel loop tile(1, 2, *) // expected-note{{active 'tile' clause defined here}}
  for(int i = 0; i < 5; ++i) {
    //expected-error@+1{{inner loops must be tightly nested inside a 'tile' clause on a 'parallel loop' construct}}
    unsigned I;
    for(int j = 0; j < 5; ++j)
      for(int k = 0; k < 5; ++k);
  }

#pragma acc parallel loop tile(1, 2, *)
  for(int i = 0; i < 5; ++i) {
    for(int j = 0; j < 5; ++j)
      for(int k = 0; k < 5; ++k)
        call();
  }

#pragma acc parallel loop tile(1, 2, *)
  // expected-error@+2{{OpenACC 'parallel loop' construct must have a terminating condition}}
  // expected-note@-2{{'parallel loop' construct is here}}
  for(int i = 0;;++i) {
  // expected-error@+2{{OpenACC 'parallel loop' construct must have a terminating condition}}
  // expected-note@-5{{'parallel loop' construct is here}}
    for(int j = 0;;++j)
  // expected-error@+2{{OpenACC 'parallel loop' construct must have a terminating condition}}
  // expected-note@-8{{'parallel loop' construct is here}}
      for(int k = 0;;++k)
        for(;;)
        call();
  }
}

void collapse_tile_depth() {
  // expected-error@+4{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+3{{active 'collapse' clause defined here}}
  // expected-error@+2{{'tile' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1{{active 'tile' clause defined here}}
#pragma acc parallel loop tile(1, 2, 3) collapse (3)
  for(int i = 0; i < 5;++i) {
    for(int j = 0; j < 5; ++j);
  }
}
