// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - 
// expected-no-diagnostics
//

int main() {
  #pragma omp teams num_teams(1:1)
  {
    // Teams region
  }

  return 0;
}
