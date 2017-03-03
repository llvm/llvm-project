// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-config suppress-null-return-paths=true -analyzer-output=text -verify %s
// expected-no-diagnostics

int *returnNull() { return 0; }
int coin();

// Use a float parameter to ensure that the value is unknown. This will create
// a cycle in the generated ExplodedGraph.
void testCycle(float i) {
  int *x = returnNull();
  int y; 
  while (i > 0) {
    x = returnNull();
    y = 2;
    i -= 1;
  }
  *x = 1; // no-warning
  y += 1;
}
