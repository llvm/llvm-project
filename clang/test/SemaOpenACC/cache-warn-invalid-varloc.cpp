// RUN: %clang_cc1 %s -fopenacc -verify

void foo() {
  int Array[5];
  // expected-warning@+1 2{{OpenACC variable in 'cache' directive was not declared outside of the associated 'loop' directive; directive has no effect}}
  #pragma acc cache(readonly:Array[1], Array[1:2])
}


void foo2() {
#pragma acc loop
  for(int i = 0; i < 5; ++i) {
    int Array[5];
    // expected-warning@+1 2{{OpenACC variable in 'cache' directive was not declared outside of the associated 'loop' directive; directive has no effect}}
    #pragma acc cache(readonly:Array[1], Array[1:2])
  }
}


template<typename T>
void foo3() {
  T Array[5];
  // expected-warning@+1 2{{OpenACC variable in 'cache' directive was not declared outside of the associated 'loop' directive; directive has no effect}}
  #pragma acc cache(readonly:Array[1], Array[1:2])
}

void Inst() {
  foo3<int>();
}
