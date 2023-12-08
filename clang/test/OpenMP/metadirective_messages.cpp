// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -verify -fopenmp -x c++ -std=c++14 -fexceptions -fcxx-exceptions %s

// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -verify -fopenmp-simd -x c++ -std=c++14 -fexceptions -fcxx-exceptions %s

void foo() {
#pragma omp metadirective // expected-error {{expected expression}}
  ;
#pragma omp metadirective when() // expected-error {{expected valid context selector in when clause}} expected-error {{expected expression}} expected-warning {{expected identifier or string literal describing a context set; set skipped}} expected-note {{context set options are: 'construct' 'device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
  ;
#pragma omp metadirective when(device{}) // expected-warning {{expected '=' after the context set name "device"; '=' assumed}} expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'kind' 'arch' 'isa'}} expected-note {{the ignored selector spans until here}} expected-error {{expected valid context selector in when clause}} expected-error {{expected expression}}
  ;
#pragma omp metadirective when(device{arch(nvptx)}) // expected-error {{missing ':' in when clause}} expected-error {{expected expression}} expected-warning {{expected '=' after the context set name "device"; '=' assumed}}
  ;
#pragma omp metadirective when(device{arch(nvptx)}: ) default() // expected-warning {{expected '=' after the context set name "device"; '=' assumed}}
  ;
#pragma omp metadirective when(device = {arch(nvptx)} : ) default(xyz) // expected-error {{expected an OpenMP directive}} expected-error {{use of undeclared identifier 'xyz'}}
  ;
#pragma omp metadirective when(device = {arch(nvptx)} : parallel default() // expected-error {{expected ',' or ')' in 'when' clause}} expected-error {{expected expression}}
  ;
#pragma omp metadirective when(device = {isa("some-unsupported-feature")} : parallel) default(single) // expected-warning {{isa trait 'some-unsupported-feature' is not known to the current target; verify the spelling or consider restricting the context selector with the 'arch' selector further}}
  ;
}
