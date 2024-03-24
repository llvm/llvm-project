#ifndef HEADER
#define HEADER
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -verify %s

#define NNN 50
int aaa[NNN];
int aaa2[NNN][NNN];

void parallel_loop() {
  #pragma omp parallel
  {
     #pragma omp loop
     for (int j = 0 ; j < NNN ; j++) {
       aaa[j] = j*NNN;
     }
   }

  #pragma omp parallel for
  for (int i = 0 ; i < NNN ; i++) {
    #pragma omp loop bind(parallel) // expected-error{{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp loop' directive into a parallel region?}}
    for (int j = 0 ; j < NNN ; j++) {
      aaa2[i][j] = i+j;
    }
  }

  #pragma omp parallel
  #pragma omp for nowait
  for (int i = 0 ; i < NNN ; i++) {
    #pragma omp loop bind(parallel) // expected-error{{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp loop' directive into a parallel region?}}
    for (int j = 0 ; j < NNN ; j++) {
      aaa2[i][j] = i+j;
    }
  }

  #pragma omp parallel for
  for (int i = 0 ; i < NNN ; i++) {
    #pragma omp nothing
    #pragma omp loop
    for (int j = 0 ; j < NNN ; j++) {
      aaa2[i][j] = i+j;
    }
  }

  #pragma omp target teams distribute parallel for
  for (int i = 0 ; i < NNN ; i++) {
    #pragma omp loop bind(parallel) // expected-error{{region cannot be closely nested inside 'target teams distribute parallel for' region; perhaps you forget to enclose 'omp loop' directive into a parallel region?}}
    for (int j = 0 ; j < NNN ; j++) {
      aaa2[i][j] = i+j;
    }
  }

  #pragma omp target parallel
  for (int i = 0 ; i < NNN ; i++) {
    #pragma omp loop bind(parallel)
    for (int j = 0 ; j < NNN ; j++) {
      aaa2[i][j] = i+j;
    }
  }

  #pragma omp parallel for
  for (int i = 0; i < 100; ++i) {
    #pragma omp loop bind(parallel) // expected-error{{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp loop' directive into a parallel region?}}
    for (int i = 0 ; i < NNN ; i++) {
      #pragma omp loop bind(parallel) // expected-error{{region cannot be closely nested inside 'loop' region; perhaps you forget to enclose 'omp loop' directive into a parallel region?}}
      for (int j = 0 ; j < NNN ; j++) {
        aaa[j] = j*NNN;
      }
    }
  }

  #pragma omp parallel
  {
     #pragma omp sections
     {
        for (int i = 0 ; i < NNN ; i++) {
          #pragma omp loop bind(parallel) // expected-error{{region cannot be closely nested inside 'sections' region; perhaps you forget to enclose 'omp loop' directive into a parallel region?}}
          for (int j = 0 ; j < NNN ; j++) {
            aaa2[i][j] = i+j;
          }
        }

        #pragma omp section
	{
          aaa[NNN-1] = NNN;
        }
     }
  }
}

void teams_loop() {
  int var1;
  int total = 0;

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

  #pragma omp target teams
  for (int i = 0 ; i < NNN ; i++) {
    #pragma omp loop bind(teams)
    for (int j = 0 ; j < NNN ; j++) {
      aaa2[i][j] = i+j;
    }
  }

  #pragma omp target teams distribute parallel for
  for (int i = 0 ; i < NNN ; i++) {
    #pragma omp loop bind(teams) // expected-error{{region cannot be closely nested inside 'target teams distribute parallel for' region; perhaps you forget to enclose 'omp loop' directive into a teams region?}}
    for (int j = 0 ; j < NNN ; j++) {
      aaa2[i][j] = i+j;
    }
  }

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

  #pragma omp teams num_teams(8) thread_limit(256)
  #pragma omp distribute parallel for dist_schedule(static, 1024) \
                                      schedule(static, 64)
  for (int i = 0; i < NNN; i++) {
    #pragma omp loop bind(teams) // expected-error{{'distribute parallel for' region; perhaps you forget to enclose 'omp loop' directive into a teams region?}}
    for (int j = 0; j < NNN; j++) {
      aaa2[i][j] = i+j;
    }
  }

  #pragma omp teams
  for (int i = 0; i < NNN; i++) {
    #pragma omp loop bind(thread)
    for (int j = 0 ; j < NNN ; j++) {
      aaa[i] = i+i*NNN;
    }
  }

  #pragma omp teams loop
  for (int i = 0; i < NNN; i++) {
    #pragma omp loop
    for (int j = 0 ; j < NNN ; j++) {
      aaa[i] = i+i*NNN;
    }
  }

  #pragma omp teams loop
  for (int i = 0; i < NNN; i++) {
    #pragma omp loop bind(teams) // expected-error{{region cannot be closely nested inside 'teams loop' region; perhaps you forget to enclose 'omp loop' directive into a teams region?}}
    for (int j = 0 ; j < NNN ; j++) {
      aaa[i] = i+i*NNN;
    }
  }
}

void thread_loop() {
  #pragma omp parallel
  for (int i = 0; i < NNN; i++) {
    #pragma omp loop bind(thread)
    for (int j = 0 ; j < NNN ; j++) {
      aaa[i] = i+i*NNN;
    }
  }

  #pragma omp teams
  for (int i = 0; i < NNN; i++) {
    #pragma omp loop bind(thread)
    for (int j = 0 ; j < NNN ; j++) {
      aaa[i] = i+i*NNN;
    }
  }
}
  
void parallel_for_with_loop_teams_bind(){
  #pragma omp parallel for
  for (int i = 0; i < NNN; i++) {
    #pragma omp loop bind(teams) // expected-error{{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp loop' directive into a teams region?}}
    for (int j = 0 ; j < NNN ; j++) {
      aaa[i] = i+i*NNN;
    }
  }
}

void orphan_loops() {
  #pragma omp loop  // expected-error{{expected 'bind' clause for 'loop' construct without an enclosing OpenMP construct}}
  for (int j = 0 ; j < NNN ; j++) {
    aaa[j] = j*NNN;
  }

  #pragma omp loop bind(parallel)
  for (int j = 0 ; j < NNN ; j++) {
    aaa[j] = j*NNN;
  }

  #pragma omp loop bind(teams)
  for (int i = 0; i < NNN; i++) {
    aaa[i] = i+i*NNN;
  }

  #pragma omp loop bind(thread)
  for (int i = 0; i < NNN; i++) {
    aaa[i] = i+i*NNN;
  }
}

int main(int argc, char *argv[]) {
  parallel_loop();
  teams_loop();
  thread_loop();
  parallel_for_with_loop_teams_bind();
  orphan_loops();
}

#endif
