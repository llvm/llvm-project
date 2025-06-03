// RUN: %clang_analyze_cc1 -Wno-format-security -Wno-pointer-to-int-cast \
// RUN:   -Wno-incompatible-library-redeclaration -verify=normaldiv %s \
// RUN:   -analyzer-checker=optin.taint.GenericTaint \
// RUN:   -analyzer-checker=core

// RUN: %clang_analyze_cc1 -Wno-format-security -Wno-pointer-to-int-cast \
// RUN:   -Wno-incompatible-library-redeclaration -verify=tainteddiv %s \
// RUN:   -analyzer-checker=optin.taint.GenericTaint \
// RUN:   -analyzer-checker=optin.taint.TaintedDiv

int getchar(void);


//If we are sure that we divide by zero
//we emit a divide by zero warning
int testDivZero(void) {
  int x = getchar(); // taint source
  if (!x)
    return 5 / x; // normaldiv-warning{{Division by zero}}
  return 8;
}

// The attacker provided value might be 0
int testDivZero2(void) {
  int x = getchar(); // taint source
  return 5 / x; // tainteddiv-warning{{Division by a tainted value}}
}

int testDivZero3(void) {
  int x = getchar(); // taint source
  if (!x)
    return 0;
  return 5 / x; // no warning
}
