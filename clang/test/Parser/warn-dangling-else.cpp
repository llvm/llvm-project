// RUN: %clang_cc1 -fsyntax-only -verify -Wdangling-else %s

void f(int a, int b, int c, int d, int e) {

  // should warn
  { if (a) if (b) d++; else e++; } // expected-warning {{add explicit braces to avoid dangling else}}
  { if (a) while (b) if (c) d++; else e++; } // expected-warning {{add explicit braces to avoid dangling else}}
  { if (a) switch (b) if (c) d++; else e++; } // expected-warning {{add explicit braces to avoid dangling else}}
  { if (a) for (;;) if (c) d++; else e++; } // expected-warning {{add explicit braces to avoid dangling else}}
  { if (a) if (b) if (d) d++; else e++; else d--; } // expected-warning {{add explicit braces to avoid dangling else}}

  if (a)
    if (b) {
      d++;
    } else e++; // expected-warning {{add explicit braces to avoid dangling else}}

  // shouldn't
  { if (a) if (b) d++; }
  { if (a) if (b) if (c) d++; }
  { if (a) if (b) d++; else e++; else d--; }
  { if (a) if (b) if (d) d++; else e++; else d--; else e--; }
  { if (a) do if (b) d++; else e++; while (c); }

  if (a) {
    if (b) d++;
    else e++;
  }

  if (a) {
    if (b) d++;
  } else e++;
}

// Somewhat more elaborate case that shouldn't warn.
class A {
 public:
  void operator<<(const char* s) {}
};

void HandleDisabledThing() {}
A GetThing() { return A(); }

#define FOO(X) \
   switch (0) default: \
     if (!(X)) \
       HandleDisabledThing(); \
     else \
       GetThing()

void f(bool cond) {
  int x = 0;
  if (cond)
    FOO(x) << "hello"; // no warning
}

