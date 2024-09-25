// RUN: %clang_cc1 %s -std=c23 -E -verify
// okay-no-diagnostics

#embed __FILE__ unrecognized
// expected-error@-1 {{unknown embed preprocessor parameter 'unrecognized'}}
#embed __FILE__ unrecognized::param
// expected-error@-1 {{unknown embed preprocessor parameter 'unrecognized::param'}}
#embed __FILE__ unrecognized::param(with, args)
// expected-error@-1 {{unknown embed preprocessor parameter 'unrecognized::param'}}
