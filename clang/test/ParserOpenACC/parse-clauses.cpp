// RUN: %clang_cc1 %s -verify -fopenacc

template<unsigned I, typename T>
void templ() {
#pragma acc loop collapse(I)
  for(int i = 0; i < 5;++i)
    for(int j = 0; j < 5; ++j)
      for(int k = 0; k < 5; ++k)
        for(int l = 0; l < 5; ++l)
          for(int m = 0; m < 5; ++m)
            for(int n = 0; n < 5; ++n)
              for(int o = 0; o < 5; ++o);

#pragma acc loop collapse(T::value)
  for(int i = 0;i < 5;++i)
    for(int j = 0; j < 5; ++j)
      for(int k = 0; k < 5; ++k)
        for(int l = 0; l < 5; ++l)
          for(int m = 0; m < 5;++m)
            for(;;)
              for(;;);

#pragma acc parallel vector_length(T::value)
  for(;;){}

#pragma acc parallel vector_length(I)
  for(;;){}

#pragma acc parallel async(T::value)
  for(;;){}

#pragma acc parallel async(I)
  for(;;){}

#pragma acc parallel async
  for(;;){}


  T t;
#pragma acc exit data delete(t)
  ;
}

struct S {
  static constexpr unsigned value = 5;
};

void use() {
  templ<7, S>();
}

// expected-error@+2{{expected ')'}}
// expected-note@+1{{to match this '('}}
#pragma acc routine(use) seq bind(NS::NSFunc)

  // expected-error@+1{{string literal with user-defined suffix cannot be used here}}
#pragma acc routine(use) seq bind("unknown udl"_UDL)

  // expected-warning@+1{{encoding prefix 'u' on an unevaluated string literal has no effect}}
#pragma acc routine(use) seq bind(u"16 bits")
void another_func();
  // expected-warning@+1{{encoding prefix 'U' on an unevaluated string literal has no effect}}
#pragma acc routine(another_func) seq bind(U"32 bits")

void AtomicIf() {
  int i, j;
  // expected-error@+1{{expected '('}}
#pragma acc atomic read if
  i = j;
#pragma acc atomic read if (true)
  i = j;
#pragma acc atomic write if (false)
  i = j + 1;

#pragma acc atomic update if (i)
  ++i;
#pragma acc atomic if (j)
  ++i;

#pragma acc atomic capture if (true)
  i = j++;
#pragma acc atomic capture if (i)
  {
    ++j;
    i = j;
  }
}
