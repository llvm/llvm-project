// RUN: %clang_cc1 -O2 -verify -emit-codegen-only %s

__attribute__((error("oh no foo"))) void foo(void);

__attribute__((error("oh no bar"))) void bar(void);

int x(void) {
  return 8 % 2 == 1;
}
void baz(void) {
  foo(); // expected-error {{call to 'foo' declared with 'error' attribute: oh no foo}}
  if (x())
    bar();
}

// FIXME: indirect call detection not yet supported.
void (*quux)(void);

void indirect(void) {
  quux = foo;
  quux();
}

// https://github.com/llvm/llvm-project/issues/146520

[[gnu::error("error please")]]
void cleaner_function(char*);

void asdf(void){
	[[gnu::cleanup(cleaner_function)]] // expected-error {{call to 'cleaner_function' declared with 'error' attribute: error please}}
	char x; 
}
