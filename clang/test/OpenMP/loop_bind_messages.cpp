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
}

void parallel_for_AND_loop_bind() {
  #pragma omp parallel for
  for (int i = 0 ; i < NNN ; i++) {
    #pragma omp loop bind(parallel) // expected-error{{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp loop' directive into a parallel region?}}
    for (int j = 0 ; j < NNN ; j++) {
      aaa2[i][j] = i+j;
    }
  }
}

void parallel_nowait() {
  #pragma omp parallel
  #pragma omp for nowait
  for (int i = 0 ; i < NNN ; i++) {
    #pragma omp loop bind(parallel) // expected-error{{region cannot be closely nested inside 'for' region; perhaps you forget to enclose 'omp loop' directive into a parallel region?}}
    for (int j = 0 ; j < NNN ; j++) {
      aaa2[i][j] = i+j;
    }
  }
}

void parallel_for_with_nothing() {
  #pragma omp parallel for
  for (int i = 0 ; i < NNN ; i++) {
    #pragma omp nothing
    #pragma omp loop // expected-error{{region cannot be closely nested inside 'parallel for' region; perhaps you forget to enclose 'omp loop' directive into a parallel region?}}
    for (int j = 0 ; j < NNN ; j++) {
      aaa2[i][j] = i+j;
    }
  }
}

void parallel_targetfor_with_loop_bind() {
  #pragma omp target teams distribute parallel for 
  for (int i = 0 ; i < NNN ; i++) {
    #pragma omp loop bind(parallel) // expected-error{{region cannot be closely nested inside 'target teams distribute parallel for' region; perhaps you forget to enclose 'omp loop' directive into a parallel region?}}
    for (int j = 0 ; j < NNN ; j++) {
      aaa2[i][j] = i+j;
    }
  }
}

void parallel_targetparallel_with_loop() {
  #pragma omp target parallel
  for (int i = 0 ; i < NNN ; i++) {
    #pragma omp loop bind(parallel)
    for (int j = 0 ; j < NNN ; j++) {
      aaa2[i][j] = i+j;
    }
  }
}

void loop_bind_AND_loop_bind() {
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
}

void parallel_with_sections_loop() {
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

void teams_targetteams_with_loop() {
  #pragma omp target teams
  for (int i = 0 ; i < NNN ; i++) {
    #pragma omp loop bind(teams)
    for (int j = 0 ; j < NNN ; j++) {
      aaa2[i][j] = i+j;
    }
  }
}

void teams_targetfor_with_loop_bind() {
  #pragma omp target teams distribute parallel for 
  for (int i = 0 ; i < NNN ; i++) {
    #pragma omp loop bind(teams) // expected-error{{region cannot be closely nested inside 'target teams distribute parallel for' region; perhaps you forget to enclose 'omp loop' directive into a teams region?}}
    for (int j = 0 ; j < NNN ; j++) {
      aaa2[i][j] = i+j;
    }
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

void teams_loop_distribute() {
  int total = 0;

  #pragma omp teams num_teams(8) thread_limit(256)
  #pragma omp distribute parallel for dist_schedule(static, 1024) \
                                      schedule(static, 64)
  for (int i = 0; i < NNN; i++) {
    #pragma omp loop bind(teams) // expected-error{{'distribute parallel for' region; perhaps you forget to enclose 'omp loop' directive into a teams region?}}
    for (int j = 0; j < NNN; j++) {
      aaa2[i][j] = i+j;
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

void teams_with_loop_thread_bind(){
  #pragma omp teams
  for (int i = 0; i < NNN; i++) {
    #pragma omp loop bind(thread)
    for (int j = 0 ; j < NNN ; j++) {
      aaa[i] = i+i*NNN;
    }
  }
}

void orphan_loop_no_bind() {
  #pragma omp loop  // expected-error{{expected 'bind' clause for 'loop' construct without an enclosing OpenMP construct}}
  for (int j = 0 ; j < NNN ; j++) {
    aaa[j] = j*NNN;
  }
}

void orphan_loop_parallel_bind() {
  #pragma omp loop bind(parallel) 
  for (int j = 0 ; j < NNN ; j++) {
    aaa[j] = j*NNN;
  }
}

void orphan_loop_teams_bind(){
  #pragma omp loop bind(teams)
  for (int i = 0; i < NNN; i++) {
    aaa[i] = i+i*NNN;
  }
}

int main(int argc, char *argv[]) {
  parallel_loop();
  parallel_for_AND_loop_bind();
  parallel_nowait();
  parallel_for_with_nothing();
  parallel_targetfor_with_loop_bind();
  parallel_targetparallel_with_loop();
  loop_bind_AND_loop_bind();
  parallel_with_sections_loop();
  teams_loop();
  teams_targetteams_with_loop();
  teams_targetfor_with_loop_bind();
  teams_loop_reduction();
  teams_loop_distribute();
  parallel_for_with_loop_teams_bind();
  teams_with_loop_thread_bind();
  orphan_loop_no_bind();
  orphan_loop_parallel_bind();
  orphan_loop_teams_bind();
}

#endif
