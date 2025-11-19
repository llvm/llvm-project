// RUN: %clang_cc1 %s -fopenacc -verify


void use() {
  int Array[5];
  int NotArray;

#pragma acc loop
  for (int i = 0; i < 5;++i) {
#pragma acc cache(Array[1])
#pragma acc cache(Array[1:2])

  // expected-error@+1{{OpenACC variable in 'cache' directive is not a valid sub-array or array element}}
#pragma acc cache(Array)
  // expected-error@+1{{OpenACC variable in 'cache' directive is not a valid sub-array or array element}}
#pragma acc cache(NotArray)
  }
}

struct S {
  int Array[5];
  int NotArray;
  int Array2D[5][5];

  void use() {
#pragma acc loop
  for (int i = 0; i < 5;++i) {
#pragma acc cache(Array[1])
#pragma acc cache(Array[1:2])
#pragma acc cache(Array2D[1][1])
#pragma acc cache(Array2D[1][1:2])

  // expected-error@+1{{OpenACC variable in 'cache' directive is not a valid sub-array or array element}}
#pragma acc cache(Array)
  // expected-error@+1{{OpenACC variable in 'cache' directive is not a valid sub-array or array element}}
#pragma acc cache(NotArray)
  }
  }
};

template<typename T>
void templ_use() {
  T Array[5];
  T NotArray;

#pragma acc loop
  for (int i = 0; i < 5;++i) {
#pragma acc cache(Array[1])
#pragma acc cache(Array[1:2])

  // expected-error@+1{{OpenACC variable in 'cache' directive is not a valid sub-array or array element}}
#pragma acc cache(Array)
  // expected-error@+1{{OpenACC variable in 'cache' directive is not a valid sub-array or array element}}
#pragma acc cache(NotArray)
  }
}

void foo() {
  templ_use<int>();
}
