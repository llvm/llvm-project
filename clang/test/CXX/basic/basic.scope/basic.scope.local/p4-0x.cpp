// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

void f() {
  int b;
  int arr[] = {1, 2, 3};

  if (bool b = true) // expected-note 2{{previous definition}}
    bool b; // expected-error {{redefinition}}
  else
    int b; // expected-error {{redefinition}}
  while (bool b = true) // expected-note {{previous definition}}
    int b; // expected-error {{redefinition}}
  for (int c; // expected-note 2{{previous definition}}
       bool c = true;) // expected-error {{redefinition}}
    double c; // expected-error {{redefinition}}
  switch (int n = 37 + 5) // expected-note {{previous definition}}
    int n; // expected-error {{redefinition}}
  for (int a : arr) // expected-note {{previous definition}}
    int a = 0; // expected-error {{redefinition}}

  if (bool b = true) { // expected-note 2{{previous definition}}
    int b; // expected-error {{redefinition}}
  } else {
    int b; // expected-error {{redefinition}}
  }
  while (bool b = true) { // expected-note {{previous definition}}
    int b; // expected-error {{redefinition}}
  }
  for (int c; // expected-note 2{{previous definition}}
       bool c = true;) { // expected-error {{redefinition}}
    double c; // expected-error {{redefinition}}
  }
  switch (int n = 37 + 5) { // expected-note {{previous definition}}
    int n; // expected-error {{redefinition}}
  }
  for (int &a : arr) { // expected-note {{previous definition}}
    int a = 0; // expected-error {{redefinition}}
  }

  if (bool b = true) {{ // expected-note {{previous definition}}
    bool b;
  }} else {
    int b; // expected-error {{redefinition}}
  }
  if (bool b = true) { // expected-note {{previous definition}}
    bool b; // expected-error {{redefinition}}
  } else {{
    int b;
  }}
  if (bool b = true) {{
    bool b;
  }} else {{
    int b;
  }}
  while (bool b = true) {{
    int b;
  }}
  for (int c; // expected-note {{previous definition}}
       bool c = true; ) {{ // expected-error {{redefinition}}
    double c;
  }}
  switch (int n = 37 + 5) {{
    int n;
  }}
  for (int &a : arr) {{
    int a = 0;
  }}
}
