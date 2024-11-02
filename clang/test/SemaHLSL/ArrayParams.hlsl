// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library %s -verify

void fn(int I[5]);     // #fn
void fn2(int I[3][3]); // #fn2

void call() {
  float F[5];
  double D[4];
  int Long[9];
  int Short[4];
  int Same[5];

  fn(F);      // expected-error{{no matching function for call to 'fn'}}
              // expected-note@#fn{{candidate function not viable: no known conversion from 'float[5]' to 'int[5]' for 1st argument}}
              
  fn(D);      // expected-error{{no matching function for call to 'fn'}}
              // expected-note@#fn{{candidate function not viable: no known conversion from 'double[4]' to 'int[5]' for 1st argument}}
  
  fn(Long);  // expected-error{{no matching function for call to 'fn'}}
             // expected-note@#fn{{candidate function not viable: no known conversion from 'int[9]' to 'int[5]' for 1st argument}}

  fn(Short); // expected-error{{no matching function for call to 'fn'}}
             // expected-note@#fn{{candidate function not viable: no known conversion from 'int[4]' to 'int[5]' for 1st argument}}
  
  fn(Same);  // totally fine, nothing to see here.
  
  fn2(Long); // expected-error{{no matching function for call to 'fn2'}}
             // expected-note@#fn2{{candidate function not viable: no known conversion from 'int[9]' to 'int[3][3]' for 1st argument}}
}
