// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

#define NUM_TEAMS 256
#define NUM_THREADS 256

#define N NUM_THREADS * NUM_TEAMS

int main() {
  int in[N], out1[N], out2[N];
  int sum1 = 0;
  int sum2 = 0;

#pragma omp target teams distribute parallel for reduction(inscan, +:sum1,sum2) map(tofrom: in, out1) num_teams(NUM_TEAMS) num_threads(NUM_THREADS)
  for(int i = 0; i < N; i++) {
    sum1 += in[i];  
    sum2 += 2*in[i];  
    #pragma omp scan inclusive(sum1,sum2) // expected-error {{multiple list items are not yet supported with the 'inclusive' or the 'exclusive' clauses that appear with the 'scan' directive}}
    out1[i] = sum1; 
    out2[i] = sum2; 
  }

#pragma omp target teams distribute parallel for reduction(inscan, +:sum1,sum2) map(tofrom: in, out1) num_teams(NUM_TEAMS) num_threads(NUM_THREADS)
  for(int i = 0; i < N; i++) {
    out1[i] = sum1; 
    out2[i] = sum2; 
    #pragma omp scan exclusive(sum1,sum2) // expected-error {{multiple list items are not yet supported with the 'inclusive' or the 'exclusive' clauses that appear with the 'scan' directive}}
    sum1 += in[i];  
    sum2 += 2*in[i];  
  }

  return 0;
}
