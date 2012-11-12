// RUN: %clang_cc1 -fsyntax-only -verify %s

// rdar: // 8125274
static int a16[];  // expected-warning {{tentative array definition assumed to have one element}}

void f16(void) {
    extern int a16[];
}


// PR10013: Scope of extern declarations extend past enclosing block
extern int PR10013_x;
int PR10013(void) {
  int *PR10013_x = 0;
  {
    extern int PR10013_x;
    extern int PR10013_x; 
  }
  
  return PR10013_x; // expected-warning{{incompatible pointer to integer conversion}}
}

