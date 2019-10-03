// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp -x c -std=c99 -fms-extensions -Wno-pragma-pack %s

// RUN: %clang_cc1 -triple=x86_64-pc-win32 -verify -fopenmp-simd -x c -std=c99 -fms-extensions -Wno-pragma-pack %s

// expected-error@+1 {{expected an OpenMP directive}}
#pragma omp declare

int foo(void);

#pragma omp declare variant // expected-error {{expected '(' after 'declare variant'}}
#pragma omp declare variant(  // expected-error {{expected expression}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare variant(foo // expected-error {{expected ')'}} expected-error {{expected 'match' clause on 'omp declare variant' directive}} expected-note {{to match this '('}}
#pragma omp declare variant(x) // expected-error {{use of undeclared identifier 'x'}}
#pragma omp declare variant(foo) // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo) // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo) xxx // expected-error {{expected 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo) match // expected-error {{expected '(' after 'match'}}
#pragma omp declare variant(foo) match( // expected-error {{expected context selector in 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo) match() // expected-error {{expected context selector in 'match' clause on 'omp declare variant' directive}}
#pragma omp declare variant(foo) match(xxx) // expected-error {{expected '=' after 'xxx' context selector set name on 'omp declare variant' directive}}
#pragma omp declare variant(foo) match(xxx=) // expected-error {{expected '{' after '='}}
#pragma omp declare variant(foo) match(xxx=yyy) // expected-error {{expected '{' after '='}}
#pragma omp declare variant(foo) match(xxx=yyy}) // expected-error {{expected '{' after '='}}
#pragma omp declare variant(foo) match(xxx={) // expected-error {{expected '}'}} expected-note {{to match this '{'}}
#pragma omp declare variant(foo) match(xxx={})
#pragma omp declare variant(foo) match(xxx={vvv})
#pragma omp declare variant(foo) match(xxx={vvv} xxx) // expected-error {{expected ','}} expected-error {{expected '=' after 'xxx' context selector set name on 'omp declare variant' directive}}
#pragma omp declare variant(foo) match(xxx={vvv}) xxx // expected-warning {{extra tokens at the end of '#pragma omp declare variant' are ignored}}
#pragma omp declare variant(foo) match(implementation={xxx}) // expected-warning {{unknown context selector in 'implementation' context selector set of 'omp declare variant' directive, ignored}}
#pragma omp declare variant(foo) match(implementation={vendor}) // expected-error {{expected '(' after 'vendor'}} expected-error {{expected vendor identifier in 'vendor' context selector of 'implementation' selector set of 'omp declare variant' directive}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare variant(foo) match(implementation={vendor(}) // expected-error {{expected vendor identifier in 'vendor' context selector of 'implementation' selector set of 'omp declare variant' directive}} expected-error {{expected ')'}} expected-note {{to match this '('}}
#pragma omp declare variant(foo) match(implementation={vendor()}) // expected-error {{expected vendor identifier in 'vendor' context selector of 'implementation' selector set of 'omp declare variant' directive}}
#pragma omp declare variant(foo) match(implementation={vendor(score ibm)}) // expected-error {{expected '(' after 'score'}} expected-warning {{missing ':' after context selector score clause - ignoring}}
#pragma omp declare variant(foo) match(implementation={vendor(score( ibm)}) // expected-error {{expected ')'}} expected-error {{use of undeclared identifier 'ibm'}} expected-error {{expected vendor identifier in 'vendor' context selector of 'implementation' selector set of 'omp declare variant' directive}} expected-warning {{missing ':' after context selector score clause - ignoring}} expected-note {{to match this '('}}
#pragma omp declare variant(foo) match(implementation={vendor(score(2 ibm)}) // expected-error 2 {{expected ')'}} expected-error {{expected vendor identifier in 'vendor' context selector of 'implementation' selector set of 'omp declare variant' directive}} expected-warning {{missing ':' after context selector score clause - ignoring}} expected-note 2 {{to match this '('}}
#pragma omp declare variant(foo) match(implementation={vendor(score(foo()) ibm)}) // expected-warning {{missing ':' after context selector score clause - ignoring}} expected-error {{expression is not an integer constant expression}}
#pragma omp declare variant(foo) match(implementation={vendor(score(5): ibm)})
int bar(void);

// expected-error@+2 {{'#pragma omp declare variant' can only be applied to functions}}
#pragma omp declare variant(foo) match(xxx={})
int a;
// expected-error@+2 {{'#pragma omp declare variant' can only be applied to functions}}
#pragma omp declare variant(foo) match(xxx={})
#pragma omp threadprivate(a)
int var;
#pragma omp threadprivate(var)

// expected-error@+2 {{expected an OpenMP directive}} expected-error@+1 {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare variant(foo) match(xxx={})
#pragma omp declare

// expected-error@+3 {{function declaration is expected after 'declare variant' directive}}
// expected-error@+1 {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare variant(foo) match(xxx={})
#pragma omp declare variant(foo) match(xxx={})
#pragma options align=packed
int main();

// expected-error@+3 {{function declaration is expected after 'declare variant' directive}}
// expected-error@+1 {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare variant(foo) match(xxx={})
#pragma omp declare variant(foo) match(xxx={})
#pragma init_seg(compiler)
int main();

// expected-error@+1 {{single declaration is expected after 'declare variant' directive}}
#pragma omp declare variant(foo) match(xxx={})
int b, c;

int no_proto();

// expected-error@+3 {{function with '#pragma omp declare variant' must have a prototype}}
// expected-note@+1 {{'#pragma omp declare variant' for function specified here}}
#pragma omp declare variant(no_proto) match(xxx={})
int no_proto_too();

int after_use_variant(void);
int after_use();
int bar() {
  return after_use();
}

// expected-warning@+1 {{'#pragma omp declare variant' cannot be applied for function after first usage; the original function might be used}}
#pragma omp declare variant(after_use_variant) match(xxx={})
int after_use(void);
#pragma omp declare variant(after_use_variant) match(xxx={})
int defined(void) { return 0; }
int defined1(void) { return 0; }
// expected-warning@+1 {{#pragma omp declare variant' cannot be applied to the function that was defined already; the original function might be used}}
#pragma omp declare variant(after_use_variant) match(xxx={})
int defined1(void);


int diff_cc_variant(void);
// expected-error@+1 {{function with '#pragma omp declare variant' has a different calling convention}}
#pragma omp declare variant(diff_cc_variant) match(xxx={})
__vectorcall int diff_cc(void);

int diff_ret_variant(void);
// expected-error@+1 {{function with '#pragma omp declare variant' has a different return type}}
#pragma omp declare variant(diff_ret_variant) match(xxx={})
void diff_ret(void);

void marked(void);
void not_marked(void);
// expected-note@+1 {{marked as 'declare variant' here}}
#pragma omp declare variant(not_marked) match(implementation={vendor(unknown)})
void marked_variant(void);
// expected-warning@+1 {{variant function in '#pragma omp declare variant' is itself marked as '#pragma omp declare variant'}}
#pragma omp declare variant(marked_variant) match(xxx={})
void marked(void);

// expected-error@+1 {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare variant
// expected-error@+1 {{function declaration is expected after 'declare variant' directive}}
#pragma omp declare variant
