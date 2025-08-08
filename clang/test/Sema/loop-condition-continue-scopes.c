// RUN: %clang_cc1 -fsyntax-only -verify %s 

void f() {
  while (({ continue; 1; })) {}
  // expected-error@-1 {{'continue' statement not in loop statement}}
  while (({ break; 1; })) {}
  // expected-error@-1 {{'break' statement not in loop or switch statement}}
  do {} while (({ break; 1; }));
  // expected-error@-1 {{'break' statement not in loop or switch statement}}
  do {} while (({ continue; 1;}));
  // expected-error@-1 {{'continue' statement not in loop statement}}
  for (({ continue; });;) {}
  // expected-error@-1 {{'continue' statement not in loop statement}}
  for (;({ continue; 1;});) {}
  // expected-error@-1 {{'continue' statement not in loop statement}}
  for (;;({ continue;})) {}
  // expected-error@-1 {{'continue' statement not in loop statement}}
  for (({ break;});;) {}
  // expected-error@-1 {{'break' statement not in loop or switch statement}}
  for (;({ break; 1;});) {}
  // expected-error@-1 {{'break' statement not in loop or switch statement}}
  for (;;({ break;})) {}
  // expected-error@-1 {{'break' statement not in loop or switch statement}}
  switch(({break;1;})){
  // expected-error@-1 {{'break' statement not in loop or switch statement}}
    case 1: break;
  }
}
