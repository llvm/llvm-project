// RUN: %clang_cc1 %s -std=c23 -E -verify

#embed __FILE__ unrecognized
// expected-error@-1 {{unknown embed preprocessor parameter 'unrecognized'}}
#embed __FILE__ unrecognized::param
// expected-error@-1 {{unknown embed preprocessor parameter 'unrecognized::param'}}
#embed __FILE__ unrecognized::param(with, args)
// expected-error@-1 {{unknown embed preprocessor parameter 'unrecognized::param'}}

// The first of these two cases was causing a failed assertion due to not being
// at the end of the directive when we finished skipping past `bla`.
#embed __FILE__ bla(2) limit(2) // expected-error {{unknown embed preprocessor parameter 'bla'}}
#embed __FILE__ limit(2) bla(2) // expected-error {{unknown embed preprocessor parameter 'bla'}}

// We intentionally only tell you about one invalid parameter at a time so that
// we can skip over invalid token soup.
#embed __FILE__ bla(2) limit(2) bork // expected-error {{unknown embed preprocessor parameter 'bla'}}
