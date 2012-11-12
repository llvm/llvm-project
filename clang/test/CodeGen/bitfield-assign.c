/* Check that the result of a bitfield assignment is properly
   truncated and does not generate a redundant load. */

/* Check that we get one load for each simple assign and two for the
   compound assign (load the old value before the add then load again
   to store back). Also check that our g0 pattern is good. */
// RUN: %clang_cc1 -triple i386-unknown-unknown -O0 -emit-llvm -o %t %s
// RUN: grep 'load ' %t | count 5
// RUN: grep "@g0" %t | count 4

// Check that we got the right value.
// RUN: %clang_cc1 -triple i386-unknown-unknown -O3 -emit-llvm -o %t %s
// RUN: grep 'load ' %t | count 0
// RUN: grep "@g0" %t | count 0

struct s0 {
  int f0 : 2;
  _Bool f1 : 1;
  unsigned f2 : 2;
};

int g0();

void f0(void) {
  struct s0 s;  
  if ((s.f0 = 3) != -1) g0();
}

void f1(void) {
  struct s0 s;  
  if ((s.f1 = 3) != 1) g0();
}

void f2(void) {
  struct s0 s;  
  if ((s.f2 = 3) != 3) g0();
}

void f3(void) {
  struct s0 s;
  // Just check this one for load counts.
  s.f0 += 3;
}

