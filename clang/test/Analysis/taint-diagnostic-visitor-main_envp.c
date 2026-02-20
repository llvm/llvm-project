// RUN: %clang_analyze_cc1 -analyzer-checker=optin.taint,core,security.ArrayBound \
// RUN: -analyzer-config assume-controlled-environment=false -analyzer-output=text -verify %s

// This file is for testing enhanced diagnostics produced by the
// GenericTaintChecker

// In an untrusted environment the cmd line arguments
// are assumed to be tainted.
int main(int argc, char *argv[], char *envp[]) { // expected-note {{Taint originated in 'argc'}}
  if (argc < 2)          // expected-note {{'argc' is >= 2}}
                         // expected-note@-1 {{Taking false branch}}
    return 1;
  int v[5] = {1, 2, 3, 4, 5};
  return v[argc]; // expected-warning {{Potential out of bound access to 'v' with tainted index}} 
                  // expected-note@-1 {{Access of 'v' with a tainted index that may be too large}}
}
