// RUN: %clang_cc1 -fsyntax-only -verify %s -Wno-unreachable-code

void test1(void) {
  { ; {  ;;}} ;;
}

void test2(void) {
  if (0) { if (1) {} } else { }

  do { } while (0); 
  
  while (0) while(0) do ; while(0);

  for ((void)0;0;(void)0)
    for (;;)
      for ((void)9;0;(void)2)
        ;
  for (int X = 0; 0; (void)0);
}

void test3(void) {
    switch (0) {
    
    case 4:
      if (0) {
    case 6: ;
      }
    default:
      ;     
  }
}

void test4(void) {
  if (0);  // expected-warning {{if statement has empty body}} expected-note {{put the semicolon on a separate line to silence this warning}}
  
  int X;  // declaration in a block.
  
foo:  if (0); // expected-warning {{if statement has empty body}} expected-note {{put the semicolon on a separate line to silence this warning}}
}

typedef int t;
void test5(void) {
  if (0);   // expected-warning {{if statement has empty body}} expected-note {{put the semicolon on a separate line to silence this warning}}

  t x = 0;

  if (0);  // expected-warning {{if statement has empty body}} expected-note {{put the semicolon on a separate line to silence this warning}}
}


void test6(void) { 
  do 
    .           // expected-error {{expected expression}}
   while (0);
}

int test7(void) {
  return 4     // expected-error {{expected ';' after return statement}}
}

void test8(void) {
  // Should not skip '}' and produce a "expected '}'" error.
  undecl // expected-error {{use of undeclared identifier 'undecl'}}
}

int test9(void) {
  int T[] = {1, 2, };

  int X;
  X = 0, // expected-error {{expected ';' after expression}}
    {
    }

  X = 0, // expected-error {{expected ';' after expression}}
  if (0)
    ;

  return 4, // expected-error {{expected ';' after return statement}}
}
