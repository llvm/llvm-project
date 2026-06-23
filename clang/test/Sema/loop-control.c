// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -x c++ %s

int pr8880_1(void) {
  int first = 1;
  for ( ; ({ if (first) { first = 0; continue; } 0; }); ) // expected-error {{'continue' statement not in loop statement}}
    return 0;
  return 1;
}

void pr8880_2(int first) {
  for ( ; ({ if (first) { first = 0; break; } 0; }); ) {} // expected-error {{'break' statement not in loop or switch statement}}
}

void pr8880_3(int first) {
  for ( ; ; (void)({ if (first) { first = 0; continue; } 0; })) {} // expected-error {{'continue' statement not in loop statement}}
}

void pr8880_4(int first) {
  for ( ; ; (void)({ if (first) { first = 0; break; } 0; })) {} // expected-error {{'break' statement not in loop or switch statement}}
}

void pr8880_5 (int first) {
  while(({ if (first) { first = 0; continue; } 0; })) {} // expected-error {{'continue' statement not in loop statement}}
}

void pr8880_6 (int first) {
  while(({ if (first) { first = 0; break; } 0; })) {} // expected-error {{'break' statement not in loop or switch statement}}
}

void pr8880_7 (int first) {
  do {} while(({ if (first) { first = 0; continue; } 0; })); // expected-error {{'continue' statement not in loop statement}}
}

void pr8880_8 (int first) {
  do {} while(({ if (first) { first = 0; break; } 0; })); // expected-error {{'break' statement not in loop or switch statement}}
}

void pr8880_10(int i) {
  for ( ; i != 10 ; i++ )
    for ( ; ; (void)({ ++i; continue; i;})) {}
}

void pr8880_11(int i) {
  for ( ; i != 10 ; i++ )
    for ( ; ; (void)({ ++i; break; i;})) {}
}

void pr8880_12(int i, int j) {
  for ( ; i != 10 ; i++ )
    for ( ; ({if (i) continue; i;}); j++) {}
}

void pr8880_13(int i, int j) {
  for ( ; i != 10 ; i++ )
    for ( ; ({if (i) break; i;}); j++) {}
}

void pr8880_14(int i) {
  for ( ; i != 10 ; i++ )
    while(({if (i) break; i;})) {}
}

void pr8880_15(int i) {
  while (--i)
    while(({if (i) continue; i;})) {}
}

void pr8880_16(int i) {
  for ( ; i != 10 ; i++ )
    do {} while(({if (i) break; i;}));
}

void pr8880_17(int i) {
  for ( ; i != 10 ; i++ )
    do {} while(({if (i) continue; i;}));
}

void pr8880_18(int x, int y) {
  while(x > 0)
    switch(({if(y) break; y;})) {
    case 2: x = 0;
    }
}

void pr8880_19(int x, int y) {
  switch(x) {
  case 1:
    switch(({if(y) break; y;})) {
    case 2: x = 0;
    }
  }
}

void pr8880_20(int x, int y) {
  switch(x) {
  case 1:
    while(({if (y) break; y;})) {}
  }
}

void pr8880_21(int x, int y) {
  switch(x) {
  case 1:
    do {} while(({if (y) break; y;}));
  }
}

void pr8880_22(int x, int y) {
  switch(x) {
  case 1:
    for ( ; ; (void)({ ++y; break; y;})) {}
  }
}

void pr8880_23(int x, int y) {
  switch(x) {
  case 1:
    for ( ; ({ ++y; break; y;}); ++y) {}
  }
}

void pr32648_1(int x, int y) {
  switch(x) {
  case 1:
    for ( ; ({ ++y; switch (y) { case 0: break; } y;}); ++y) {} // no warning
  }
}

void pr32648_2(int x, int y) {
  while(x) {
    for ( ; ({ ++y; switch (y) { case 0: continue; } y;}); ++y) {}
  }
}

void pr32648_3(int x, int y) {
  switch(x) {
  case 1:
    for ( ; ({ ++y; for (; y; y++) { break; } y;}); ++y) {} // no warning
  }
}

void pr32648_4(int x, int y) {
  switch(x) {
  case 1:
    for ( ; ({ ++y; for (({ break; }); y; y++) { } y;}); ++y) {}
  }
}

void pr32648_5(int x, int y) {
  switch(x) {
  case 1:
    for ( ; ({ ++y; while (({ break; y; })) {} y;}); ++y) {}
  }
}

void pr32648_6(int x, int y) {
  switch(x) {
  case 1:
    for ( ; ({ ++y; do {} while (({ break; y; })); y;}); ++y) {}
  }
}

void pr32648_7(int x, int y) {
  switch(x) {
  case 1:
    for ( ; ({ ++y; do { break; } while (y); y;}); ++y) {} // no warning
  }
}
