// RUN: %clang_cc1 %s -std=c23 -E -verify
// RUN: %clang_cc1 %s -std=c23 -E -Wno-unknown-directive-parameters -verify=okay
// okay-no-diagnostics

#embed __FILE__ unrecognized
// expected-warning@-1 {{unknown embed preprocessor parameter 'unrecognized' ignored}}
#embed __FILE__ unrecognized::param
// expected-warning@-1 {{unknown embed preprocessor parameter 'unrecognized::param' ignored}}
#embed __FILE__ unrecognized::param(with, args)
// expected-warning@-1 {{unknown embed preprocessor parameter 'unrecognized::param' ignored}}
