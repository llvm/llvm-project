// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=51 %s

// Test 4: Reuse with overloaded operator= (should error)
struct Iterator {
  int value;
  Iterator& operator=(int v) { value = v; return *this; }
  bool operator<(int n) const { return value < n; }
  Iterator& operator++() { ++value; return *this; }                           
}; 

Iterator i;
extern int &dim;
auto test4() {  
  #pragma omp parallel for collapse(2)                                          
    for (i = 0; i < dim; ++i) {                              
      // expected-error@+1{{loop iteration variable 'i' cannot be reused in a nested loop of a collapsed loop nest}}                                        
      for (i = 0; i < 10; ++i) {                                      
        int dummy;                                                              
      }           
    }                                                                           
  } 

auto test5() {
#pragma omp parallel for collapse(2)                                          
  for (i = 0; i < dim; ++i) {                              
    // expected-error@+1{{loop iteration variable 'i' cannot be reused in a nested loop of a collapsed loop nest}}                                        
    for (i = 0; i < 10; ++i) {                             
      int dummy;                                                              
    }                                                                         
  }                                                                           
} 
