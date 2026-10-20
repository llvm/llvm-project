// RUN: %clang_cc1 %s -verify -fopenacc
//
// expected-error@+1{{OpenACC construct 'loop' cannot be used here; it can only be used in a statement context}}
#pragma acc loop

// expected-error@+1{{OpenACC construct 'loop' cannot be used here; it can only be used in a statement context}}
#pragma acc loop
int foo;

struct S {
// expected-error@+1{{OpenACC construct 'loop' cannot be used here; it can only be used in a statement context}}
#pragma acc loop
  int i;

  void mem_func() {
  // expected-error@+3{{OpenACC 'loop' construct can only be applied to a 'for' loop}}
  // expected-note@+1{{'loop' construct is here}}
#pragma acc loop
    int foo;

  // expected-error@+3{{OpenACC 'loop' construct can only be applied to a 'for' loop}}
  // expected-note@+1{{'loop' construct is here}}
#pragma acc loop
    while(0);

  // expected-error@+3{{OpenACC 'loop' construct can only be applied to a 'for' loop}}
  // expected-note@+1{{'loop' construct is here}}
#pragma acc loop
    do{}while(0);

  // expected-error@+3{{OpenACC 'loop' construct can only be applied to a 'for' loop}}
  // expected-note@+1{{'loop' construct is here}}
#pragma acc loop
    {}

#pragma acc loop
    for(int i = 0; i < 6; ++i);

    int array[5];

#pragma acc loop
    for(auto X : array){}
}
};

template<typename T>
void templ_func() {
  // expected-error@+3{{OpenACC 'loop' construct can only be applied to a 'for' loop}}
  // expected-note@+1{{'loop' construct is here}}
#pragma acc loop
  int foo;

  // expected-error@+3{{OpenACC 'loop' construct can only be applied to a 'for' loop}}
  // expected-note@+1{{'loop' construct is here}}
#pragma acc loop
  while(T{});

  // expected-error@+3{{OpenACC 'loop' construct can only be applied to a 'for' loop}}
  // expected-note@+1{{'loop' construct is here}}
#pragma acc loop
  do{}while(0);

  // expected-error@+3{{OpenACC 'loop' construct can only be applied to a 'for' loop}}
  // expected-note@+1{{'loop' construct is here}}
#pragma acc loop
  {}

#pragma acc loop
  for(T i = 0; i < 1; ++i);

  T array[5];

#pragma acc loop
  for(auto X : array){}
}

void use() {
  templ_func<int>();
}

