// RUN: %clang_cc1 -verify -fopenmp -fopenmp-target-xteam-scan %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-target-xteam-scan %s -Wuninitialized

// RUN: %clang_cc1 -verify=missing-flag,expected -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify=missing-flag,expected -fopenmp-simd %s -Wuninitialized

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
    // missing-flag-error@+2 {{'scan' directive is not supported inside target regions. Use flag '-fopenmp-target-xteam-scan' to enable it}}
    // expected-error@+1 {{multiple list items are not yet supported with the 'inclusive' or the 'exclusive' clauses that appear with the 'scan' directive}}
    #pragma omp scan inclusive(sum1,sum2)
    out1[i] = sum1; 
    out2[i] = sum2; 
  }

#pragma omp target teams distribute parallel for reduction(inscan, +:sum1,sum2) map(tofrom: in, out1) num_teams(NUM_TEAMS) num_threads(NUM_THREADS)
  for(int i = 0; i < N; i++) {
    out1[i] = sum1; 
    out2[i] = sum2; 
    // missing-flag-error@+2 {{'scan' directive is not supported inside target regions. Use flag '-fopenmp-target-xteam-scan' to enable it}}
    // expected-error@+1 {{multiple list items are not yet supported with the 'inclusive' or the 'exclusive' clauses that appear with the 'scan' directive}}
    #pragma omp scan exclusive(sum1,sum2) 
    sum1 += in[i];  
    sum2 += 2*in[i];  
  }


  return 0;
}
