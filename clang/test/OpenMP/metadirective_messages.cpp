// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -verify -fopenmp -x c++ -std=c++14 -fexceptions -fcxx-exceptions %s

// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -verify -fopenmp-simd -x c++ -std=c++14 -fexceptions -fcxx-exceptions %s

// RUN: %clang_cc1 -triple=x86_64-pc-linux-gnu -verify=expected,omp52 -fopenmp -fopenmp-version=52 -ferror-limit 100 -o - %s -Wuninitialized

void foo() {
  #if _OPENMP >= 202111
    #pragma omp metadirective // omp52-error {{expected expression}}
      ;
    #pragma omp metadirective when() // omp52-error {{expected valid context selector in when clause}} expected-error {{expected expression}} expected-warning {{expected identifier or string literal describing a context set; set skipped}} expected-note {{context set options are: 'construct' 'device' 'target_device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
      ;
    #pragma omp metadirective when(device{}) // omp52-warning {{expected '=' after the context set name "device"; '=' assumed}} expected-warning {{expected identifier or string literal describing a context selector; selector skipped}} expected-note {{context selector options are: 'kind' 'arch' 'isa'}} expected-note {{the ignored selector spans until here}} expected-error {{expected valid context selector in when clause}} expected-error {{expected expression}}
      ;
    #pragma omp metadirective when(device{arch(nvptx)}) // omp52-error {{missing ':' in when clause}} expected-error {{expected expression}} expected-warning {{expected '=' after the context set name "device"; '=' assumed}}
      ;
    #pragma omp metadirective when(device{arch(nvptx)}: ) otherwise() // omp52-warning {{expected '=' after the context set name "device"; '=' assumed}}
      ;
    #pragma omp metadirective when(device = {arch(nvptx)} : ) otherwise(xyz) // omp52-error {{expected an OpenMP directive}} expected-error {{use of undeclared identifier 'xyz'}}
      ;
    #pragma omp metadirective when(device = {arch(nvptx)} : parallel otherwise() // omp52-error {{expected ',' or ')' in 'when' clause}} expected-error {{expected expression}}
      ;
    #pragma omp metadirective when(device = {isa("some-unsupported-feature")} : parallel) otherwise(single) // omp52-warning {{isa trait 'some-unsupported-feature' is not known to the current target; verify the spelling or consider restricting the context selector with the 'arch' selector further}}
      ;
    #pragma omp metadirective when(device = {arch(nvptx)} : parallel) default() // omp52-warning {{'default' clause for 'metadirective' is deprecated; use 'otherwise' instead}}
      ;
    #pragma omp metadirective when(device = {arch(nvptx)} : parallel) xyz() //omp52-error {{expected at least one clause on '#pragma omp metadirective' directive}} 
      ;
  #else
    #pragma omp metadirective // expected-error {{expected expression}}
      ;
    #pragma omp metadirective when() // expected-error {{expected valid context selector in when clause}} expected-error {{expected expression}} expected-warning {{expected identifier or string literal describing a context set; set skipped}} expected-note {{context set options are: 'construct' 'device' 'target_device' 'implementation' 'user'}} expected-note {{the ignored set spans until here}}
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
    #pragma omp metadirective when(device = {isa("some-unsupported-feature")} : parallel) otherwise(single) // expected-warning {{isa trait 'some-unsupported-feature' is not known to the current target; verify the spelling or consider restricting the context selector with the 'arch' selector further}} //expected-error{{unexpected OpenMP clause 'otherwise' in directive '#pragma omp metadirective'}}
      ;
    #pragma omp metadirective when(device = {arch(nvptx)} : parallel) xyz() //expected-error {{expected at least one clause on '#pragma omp metadirective' directive}} 
      ;
  #endif
  }

namespace GH139665 {
void f(){
#pragma omp metadirective( // expected-error {{expected at least one clause on '#pragma omp metadirective' directive}}
}

void g() {
#pragma omp metadirective align // expected-error {{expected '(' after 'align'}}
}
} // namespace GH139665
