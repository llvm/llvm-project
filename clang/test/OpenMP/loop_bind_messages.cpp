#ifndef HEADER
#define HEADER
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -verify %s

#define NNN 50
int aaa[NNN];

void parallel_loop() {
  #pragma omp parallel
  {
     #pragma omp loop
     for (int j = 0 ; j < NNN ; j++) {
       aaa[j] = j*NNN;
     }
   }
}

void teams_loop() {
  int var1, var2;

  #pragma omp teams
  {
     #pragma omp loop bind(teams)
     for (int j = 0 ; j < NNN ; j++) {
       aaa[j] = j*NNN;
     }

     #pragma omp loop bind(teams) collapse(2) private(var1)
     for (int i = 0 ; i < 3 ; i++) {
       for (int j = 0 ; j < NNN ; j++) {
         var1 += aaa[j];
       }
     }
   }
}

void orphan_loop_with_bind() {
  #pragma omp loop bind(parallel) 
  for (int j = 0 ; j < NNN ; j++) {
    aaa[j] = j*NNN;
  }
}

void orphan_loop_no_bind() {
  #pragma omp loop  // expected-error{{expected 'bind' clause for 'loop' construct without an enclosing OpenMP construct}}
  for (int j = 0 ; j < NNN ; j++) {
    aaa[j] = j*NNN;
  }
}

void teams_loop_reduction() {
  int total = 0;

  #pragma omp teams
  {
     #pragma omp loop bind(teams)
     for (int j = 0 ; j < NNN ; j++) {
       aaa[j] = j*NNN;
     }

     #pragma omp loop bind(teams) reduction(+:total) // expected-error{{'reduction' clause not allowed with '#pragma omp loop bind(teams)'}}
     for (int j = 0 ; j < NNN ; j++) {
       total+=aaa[j];
     }
   }
}

int main(int argc, char *argv[]) {
  parallel_loop();
  teams_loop();
  orphan_loop_with_bind();
  orphan_loop_no_bind();
  teams_loop_reduction();
}

#endif
