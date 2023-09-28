// RUN: %clang_cc1 %s -E -CC -verify

#embed __FILE__ unrecognized
// expected-warning@-1 {{unknown embed preprocessor parameter 'unrecognized' ignored}}
#embed __FILE__ unrecognized::param
// expected-warning@-1 {{unknown embed preprocessor parameter 'unrecognized::param' ignored}}
#embed __FILE__ unrecognized::param(with, args)
// expected-warning@-1 {{unknown embed preprocessor parameter 'unrecognized::param' ignored}}
