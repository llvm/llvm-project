// RUN: %clang_cc1 -verify -fopenmp -ferror-limit 100 %s -Wuninitialized
// RUN: %clang_cc1 -verify -fopenmp-simd -ferror-limit 100 %s -Wuninitialized

int main(int argc, char **argv) {
  int len = 16;
  int count = 8;
  int divisor = 2;
  double *data;
  
  // Valid strided array sections with variable count expressions (FROM)
  #pragma omp target update from(data[0:count:2]) // OK - variable count
  
  #pragma omp target update from(data[0:len/2:2]) // OK - count expression
  
  #pragma omp target update from(data[0:len-4:3]) // OK - count with subtraction
  
  #pragma omp target update from(data[1:(len+1)/2:2]) // OK - complex count expression
  
  #pragma omp target update from(data[0:count*2:3]) // OK - count multiplication
  
  #pragma omp target update from(data[2:len%divisor:2]) // OK - count with modulo
  
  // Variable count with stride = 1 (contiguous)
  #pragma omp target update from(data[0:count]) // OK - variable count, implicit stride
  
  #pragma omp target update from(data[0:len/divisor]) // OK - expression count, implicit stride
  
  // Invalid stride expressions with variable count
  #pragma omp target update from(data[0:count:0]) // expected-error {{section stride is evaluated to a non-positive value 0}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  #pragma omp target update from(data[0:len/2:-1]) // expected-error {{section stride is evaluated to a non-positive value -1}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  #pragma omp target update from(data[1:count:-2]) // expected-error {{section stride is evaluated to a non-positive value -2}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  // Valid strided array sections with variable count expressions (TO)
  #pragma omp target update to(data[0:count:2]) // OK
  
  #pragma omp target update to(data[0:len/2:2]) // OK
  
  #pragma omp target update to(data[0:len-4:3]) // OK
  
  // Invalid stride with TO
  #pragma omp target update to(data[0:count:0]) // expected-error {{section stride is evaluated to a non-positive value 0}} expected-error {{expected at least one 'to' clause or 'from' clause specified to '#pragma omp target update'}}
  
  return 0;
}
