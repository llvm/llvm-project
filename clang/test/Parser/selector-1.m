// RUN: %clang_cc1 -fsyntax-only -verify %s 
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify %s 
// expected-no-diagnostics

int main(void) {
  SEL s = @selector(retain);
  SEL s1 = @selector(meth1:);
  SEL s2 = @selector(retainArgument::);
  SEL s3 = @selector(retainArgument:::::);
  SEL s4 = @selector(retainArgument:with:);
  SEL s5 = @selector(meth1:with:with:);
  SEL s6 = @selector(getEnum:enum:bool:);
  SEL s7 = @selector(char:float:double:unsigned:short:long:);
  SEL s9 = @selector(:enum:bool:);
  
  (void) @selector(foo:);
  (void) @selector(foo::);
  (void) @selector(foo:::);
  (void) @selector(foo::::);
}
