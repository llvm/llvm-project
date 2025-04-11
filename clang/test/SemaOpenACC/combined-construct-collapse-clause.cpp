// RUN: %clang_cc1 %s -fopenacc -verify

template<unsigned Val>
void depth_too_high_templ() {
  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc parallel loop collapse(Val)
  for(unsigned i = 0; i < 5; ++i)
    for(unsigned j = 0; j < 5; ++j);
}
constexpr int three() { return 3; }
struct ConvertsThree{
  constexpr ConvertsThree(){};

  constexpr operator int(){ return 3; }
};

void depth_too_high() {
  depth_too_high_templ<3>(); // expected-note{{in instantiation of function template specialization}}

  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc serial loop collapse(3)
  for(unsigned i = 0; i < 5; ++i)
    for(unsigned j = 0; j < 5; ++j);

  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc kernels loop collapse(three())
  for(unsigned i = 0; i < 5; ++i)
    for(unsigned j = 0; j < 5; ++j);

  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc parallel loop collapse(ConvertsThree{})
  for(unsigned i = 0; i < 5; ++i)
    for(unsigned j = 0; j < 5; ++j);
}

template<typename T, unsigned Three>
void not_single_loop_templ() {
  T Arr[5];
  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1 2{{active 'collapse' clause defined here}}
#pragma acc parallel loop collapse(3)
  for(auto x : Arr) {
    for(auto y : Arr){
      do{}while(true); // expected-error{{do loop cannot appear in intervening code of a 'parallel loop' with a 'collapse' clause}}
    }
  }

  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1 2{{active 'collapse' clause defined here}}
#pragma acc serial loop collapse(Three)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {
      do{}while(true); // expected-error{{do loop cannot appear in intervening code of a 'serial loop' with a 'collapse' clause}}
    }
  }

#pragma acc kernels loop collapse(Three)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {
      for(unsigned k = 0; k < 5;++k) {
        do{}while(true);
      }
    }
  }
  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1 2{{active 'collapse' clause defined here}}
#pragma acc parallel loop collapse(Three)
  for(auto x : Arr) {
    for(auto y: Arr) {
      do{}while(true); // expected-error{{do loop cannot appear in intervening code of a 'parallel loop' with a 'collapse' clause}}
    }
  }

#pragma acc serial loop collapse(Three)
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
#pragma acc kernels loop collapse(3)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {
      for(unsigned k = 0; k < 5;++k);
    }
    while(true); // expected-error{{while loop cannot appear in intervening code of a 'kernels loop' with a 'collapse' clause}}
  }

  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc parallel loop collapse(3)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {
      for(unsigned k = 0; k < 5;++k);
    }
    do{}while(true); // expected-error{{do loop cannot appear in intervening code of a 'parallel loop' with a 'collapse' clause}}
  }

  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1 2{{active 'collapse' clause defined here}}
#pragma acc serial loop collapse(3)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {
      while(true); // expected-error{{while loop cannot appear in intervening code of a 'serial loop' with a 'collapse' clause}}
    }
  }
  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1 2{{active 'collapse' clause defined here}}
#pragma acc kernels loop collapse(3)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {
      do{}while(true); // expected-error{{do loop cannot appear in intervening code of a 'kernels loop' with a 'collapse' clause}}
    }
  }

#pragma acc parallel loop collapse(2)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {
      do{}while(true);
    }
  }
#pragma acc serial loop collapse(2)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {
      while(true);
    }
  }

  int Arr[5];
  // expected-error@+2{{'collapse' clause specifies a loop count greater than the number of available loops}}
  // expected-note@+1 2{{active 'collapse' clause defined here}}
#pragma acc kernels loop collapse(3)
  for(auto x : Arr) {
    for(auto y : Arr){
      do{}while(true); // expected-error{{do loop cannot appear in intervening code of a 'kernels loop' with a 'collapse' clause}}
    }
  }

  // expected-note@+1 {{active 'collapse' clause defined here}}
#pragma acc parallel loop collapse(3)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {
      for(unsigned k = 0; k < 5;++k);
    }
    // expected-error@+1{{more than one for-loop in a loop associated with OpenACC 'parallel loop' construct with a 'collapse' clause}}
      for(unsigned k = 0; k < 5;++k);
  }

  // expected-note@+1 {{active 'collapse' clause defined here}}
#pragma acc serial loop collapse(3)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {
      for(unsigned k = 0; k < 5;++k);
    // expected-error@+1{{more than one for-loop in a loop associated with OpenACC 'serial loop' construct with a 'collapse' clause}}
      for(unsigned k = 0; k < 5;++k);
    }
  }

  for(unsigned k = 0; k < 5;++k);
#pragma acc kernels loop collapse(3)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {
      for(unsigned k = 0; k < 5;++k);
    }
  }
}

template<unsigned Two, unsigned Three>
void no_other_directives() {
#pragma acc parallel loop collapse(Two)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {// last loop associated with the top level.
    // expected-error@+1{{'collapse' clause specifies a loop count greater than the number of available loops}}
#pragma acc serial loop collapse(Three) // expected-note 2{{active 'collapse' clause defined here}}
      for(unsigned k = 0; k < 6;++k) {
        for(unsigned l = 0; l < 5; ++l) {
    // expected-error@+1{{OpenACC 'serial' construct cannot appear in intervening code of a 'serial loop' with a 'collapse' clause}}
#pragma acc serial
          ;
        }
      }
    }
  }
#pragma acc kernels loop collapse(Two)// expected-note{{active 'collapse' clause defined here}}
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {// last loop associated with the top level.
#pragma acc parallel loop collapse(Three)
      for(unsigned k = 0; k < 6;++k) {
        for(unsigned l = 0; l < 5; ++l) {
          for(unsigned m = 0; m < 5; ++m);
        }
      }
    }
    // expected-error@+1{{OpenACC 'serial' construct cannot appear in intervening code of a 'kernels loop' with a 'collapse' clause}}
#pragma acc serial
          ;
  }
}

void no_other_directives() {
  no_other_directives<2,3>(); // expected-note{{in instantiation of function template specialization}}

  // Ok, not inside the intervening list
#pragma acc serial loop collapse(2)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {
    // expected-error@+1{{OpenACC 'data' construct must have at least one 'copy', 'copyin', 'copyout', 'create', 'no_create', 'present', 'deviceptr', 'attach' or 'default' clause}}
#pragma acc data
      ;
    }
  }
  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc kernels loop collapse(2)
  for(unsigned i = 0; i < 5; ++i) {
    // expected-error@+2{{OpenACC 'data' construct must have at least one 'copy', 'copyin', 'copyout', 'create', 'no_create', 'present', 'deviceptr', 'attach' or 'default' clause}}
    // expected-error@+1{{OpenACC 'data' construct cannot appear in intervening code of a 'kernels loop' with a 'collapse' clause}}
#pragma acc data
    for(unsigned j = 0; j < 5; ++j) {
    }
  }
}

void call();

template<unsigned Two>
void intervening_without_force_templ() {
  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc parallel loop collapse(2)
  for(unsigned i = 0; i < 5; ++i) {
    // expected-error@+1{{inner loops must be tightly nested inside a 'collapse' clause on a 'parallel loop' construct}}
    call();
    for(unsigned j = 0; j < 5; ++j);
  }

  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc serial loop collapse(Two)
  for(unsigned i = 0; i < 5; ++i) {
    // expected-error@+1{{inner loops must be tightly nested inside a 'collapse' clause on a 'serial loop' construct}}
    call();
    for(unsigned j = 0; j < 5; ++j);
  }

  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc kernels loop collapse(2)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j);
    // expected-error@+1{{inner loops must be tightly nested inside a 'collapse' clause on a 'kernels loop' construct}}
    call();
  }

#pragma acc parallel loop collapse(force:2)
  for(unsigned i = 0; i < 5; ++i) {
    call();
    for(unsigned j = 0; j < 5; ++j);
  }

#pragma acc parallel loop collapse(force:Two)
  for(unsigned i = 0; i < 5; ++i) {
    call();
    for(unsigned j = 0; j < 5; ++j);
  }


#pragma acc parallel loop collapse(force:2)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j);
    call();
  }

#pragma acc parallel loop collapse(force:Two)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j);
    call();
  }

#pragma acc parallel loop collapse(Two)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {
    call();
    }
  }

#pragma acc parallel loop collapse(Two)
  for(unsigned i = 0; i < 5; ++i) {
    {
      {
        for(unsigned j = 0; j < 5; ++j) {
          call();
        }
      }
    }
  }

#pragma acc parallel loop collapse(force:Two)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {
    call();
    }
  }

  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc parallel loop collapse(Two)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j);
    // expected-error@+1{{inner loops must be tightly nested inside a 'collapse' clause on a 'parallel loop' construct}}
    call();
  }

#pragma acc parallel loop collapse(2)
  // expected-error@+2{{OpenACC 'parallel loop' construct must have a terminating condition}}
  // expected-note@-2{{'parallel loop' construct is here}}
  for(int i = 0;;++i)
  // expected-error@+2{{OpenACC 'parallel loop' construct must have a terminating condition}}
  // expected-note@-5{{'parallel loop' construct is here}}
    for(int j = 0;;++j)
      for(;;);
}

void intervening_without_force() {
  intervening_without_force_templ<2>(); // expected-note{{in instantiation of function template specialization}}
  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc parallel loop collapse(2)
  for(unsigned i = 0; i < 5; ++i) {
    // expected-error@+1{{inner loops must be tightly nested inside a 'collapse' clause on a 'parallel loop' construct}}
    call();
    for(unsigned j = 0; j < 5; ++j);
  }

  // expected-note@+1{{active 'collapse' clause defined here}}
#pragma acc parallel loop collapse(2)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j);
    // expected-error@+1{{inner loops must be tightly nested inside a 'collapse' clause on a 'parallel loop' construct}}
    call();
  }

  // The below two are fine, as they use the 'force' tag.
#pragma acc parallel loop collapse(force:2)
  for(unsigned i = 0; i < 5; ++i) {
    call();
    for(unsigned j = 0; j < 5; ++j);
  }

#pragma acc parallel loop collapse(force:2)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j);
    call();
  }

#pragma acc parallel loop collapse(2)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {
    call();
    }
  }
#pragma acc parallel loop collapse(2)
  for(unsigned i = 0; i < 5; ++i) {
    {
      {
        for(unsigned j = 0; j < 5; ++j) {
          call();
        }
      }
    }
  }

#pragma acc parallel loop collapse(force:2)
  for(unsigned i = 0; i < 5; ++i) {
    for(unsigned j = 0; j < 5; ++j) {
    call();
    }
  }

#pragma acc parallel loop collapse(2)
  // expected-error@+2{{OpenACC 'parallel loop' construct must have a terminating condition}}
  // expected-note@-2{{'parallel loop' construct is here}}
  for(int i = 0;;++i)
  // expected-error@+2{{OpenACC 'parallel loop' construct must have a terminating condition}}
  // expected-note@-5{{'parallel loop' construct is here}}
    for(int j = 0;;++j)
      for(;;);
}

