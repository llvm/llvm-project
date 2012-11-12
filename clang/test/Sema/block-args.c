// RUN: %clang_cc1 %s -fsyntax-only -verify -fblocks

void take(void*);

void test() {
  take(^(int x){});
  take(^(int x, int y){});
  take(^(int x, int y){});
  take(^(int x,      // expected-note {{previous declaration is here}}
         int x){});  // expected-error {{redefinition of parameter 'x'}}


  take(^(int x) { return x+1; });

  int (^CP)(int) = ^(int x) { return x*x; };
  take(CP);

  int arg;
  ^{return 1;}();
  ^{return 2;}(arg); // expected-error {{too many arguments to block call}}
  ^(void){return 3;}(1); // expected-error {{too many arguments to block call}}
  ^(){return 4;}(arg); // expected-error {{too many arguments to block call}}
  ^(int x, ...){return 5;}(arg, arg);   // Explicit varargs, ok.
}

int main(int argc, char** argv) {
  ^(int argCount) {
    argCount = 3;
  }(argc);
}

// radar 7528255
void f0() {
  ^(int, double d, char) {}(1, 1.34, 'a'); // expected-error {{parameter name omitted}} \
				 	   // expected-error {{parameter name omitted}}
}

// rdar://problem/8962770
void test4() {
  int (^f)() = ^((x)) { }; // expected-error {{expected ')'}} expected-warning {{type specifier missing}} expected-note {{to match this}}
}

// rdar://problem/9170609
void test5_helper(void (^)(int, int[*]));
void test5(void) {
  test5_helper(^(int n, int array[n]) {});
}
