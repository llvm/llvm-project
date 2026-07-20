// RUN: %clang_cc1 -verify -fsyntax-only -std=c90 -DPRE_C99 %s
// RUN: %clang_cc1 -verify -fsyntax-only -std=c99 %s
// RUN: %clang_cc1 -verify -fsyntax-only -std=c2y %s
// RUN: %clang_cc1 -verify -fsyntax-only -x c++ -std=c++20 %s

int arr[5];

void err(int q) {
  while (({ continue; 1; })) {} // expected-error {{'continue' statement not in loop statement}}
  while (({ break; 1; })) {} // expected-error {{'break' statement not in loop or switch statement}}

  do {} while (({ continue; 1; })); // expected-error {{'continue' statement not in loop statement}}
  do {} while (({ break; 1; })); // expected-error {{'break' statement not in loop or switch statement}}

  for (({continue;});;) {} // expected-error {{'continue' statement not in loop statement}}
  for (({break;});;) {} // expected-error {{'break' statement not in loop or switch statement}}
  for (; ({continue; 1;});) {} // expected-error {{'continue' statement not in loop statement}}
  for (; ({break; 1;});) {} // expected-error {{'break' statement not in loop or switch statement}}
  for (;;({continue;})) {} // expected-error {{'continue' statement not in loop statement}}
  for (;;({break;})) {} // expected-error {{'break' statement not in loop or switch statement}}

#ifndef PRE_C99
  for (int x = ({continue; 1;});;) {} // expected-error {{'continue' statement not in loop statement}}
  for (int x = ({break; 1;});;) {} // expected-error {{'break' statement not in loop or switch statement}}
#endif

#if __cplusplus
  for (; int x = ({continue; 1;});) {} // expected-error {{'continue' statement not in loop statement}}
  for (; int x = ({break; 1;});) {} // expected-error {{'break' statement not in loop or switch statement}}
  for (({continue;}); int x : arr) {} // expected-error {{'continue' statement not in loop statement}}
  for (({break;}); int x : arr) {} // expected-error {{'break' statement not in loop or switch statement}}
  for (int y = ({continue; 5;}); int x : arr) {} // expected-error {{'continue' statement not in loop statement}}
  for (int y = ({break; 5;}); int x : arr) {} // expected-error {{'break' statement not in loop or switch statement}}
  for (int x : *({ continue; &arr; })) {} // expected-error {{'continue' statement not in loop statement}}
  for (int x : *({ break; &arr; })) {} // expected-error {{'break' statement not in loop or switch statement}}
#endif

  switch (({continue; q;})) {} // expected-error {{'continue' statement not in loop statement}}
  switch (({break; q;})) {} // expected-error {{'break' statement not in loop or switch statement}}
}

void in_outer_loop(int q) {
  for (;;) {
    while (({ continue; 1; })) {}
    while (({ break; 1; })) {}

    do {} while (({ continue; 1; }));
    do {} while (({ break; 1; }));

    for (({continue;});;) {}
    for (({break;});;) {}
    for (; ({continue; 1;});) {}
    for (; ({break; 1;});) {}
    for (;;({continue;})) {}
    for (;;({break;})) {}

#ifndef PRE_C99
    for (int x = ({continue; 1;});;) {}
    for (int x = ({break; 1;});;) {}
#endif

#if __cplusplus
    for (; int x = ({continue; 1;});) {}
    for (; int x = ({break; 1;});) {}
    for (({continue;}); int x : arr) {}
    for (({break;}); int x : arr) {}
    for (int y = ({continue; 5;}); int x : arr) {}
    for (int y = ({break; 5;}); int x : arr) {}
    for (int x : *({ continue; &arr; })) {}
    for (int x : *({ break; &arr; })) {}
#endif

    switch (({continue; q;})) {}
    switch (({break; q;})) {}
  }
}

void in_outer_switch(int y) {
  switch (y) {
    default: {
      while (({ continue; 1; })) {} // expected-error {{'continue' statement not in loop statement}}
      while (({ break; 1; })) {}

      do {} while (({ continue; 1; })); // expected-error {{'continue' statement not in loop statement}}
      do {} while (({ break; 1; }));

      for (({continue;});;) {} // expected-error {{'continue' statement not in loop statement}}
      for (({break;});;) {}
      for (; ({continue; 1;});) {} // expected-error {{'continue' statement not in loop statement}}
      for (; ({break; 1;});) {}
      for (;;({continue;})) {} // expected-error {{'continue' statement not in loop statement}}
      for (;;({break;})) {}

#ifndef PRE_C99
      for (int x = ({continue; 1;});;) {} // expected-error {{'continue' statement not in loop statement}}
      for (int x = ({break; 1;});;) {}
#endif

#if __cplusplus
      for (; int x = ({continue; 1;});) {} // expected-error {{'continue' statement not in loop statement}}
      for (; int x = ({break; 1;});) {}
      for (({continue;}); int x : arr) {} // expected-error {{'continue' statement not in loop statement}}
      for (({break;}); int x : arr) {}
      for (int y = ({continue; 5;}); int x : arr) {} // expected-error {{'continue' statement not in loop statement}}
      for (int y = ({break; 5;}); int x : arr) {}
      for (int x : *({ continue; &arr; })) {} // expected-error {{'continue' statement not in loop statement}}
      for (int x : *({ break; &arr; })) {}
#endif

      switch (({continue; y;})) {} // expected-error {{'continue' statement not in loop statement}}
      switch (({break; y;})) {}
    }
  }
}
