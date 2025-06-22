// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify -triple amdgcn -Wconvergence %s

#define __device__ __attribute__((device))

__device__ void foo() __attribute__((convergent));

__device__ int jump_into_nest(int cond) {
  int i = 0;

  if (cond > 1) {
    goto jumptarget; // #GOTO11
  }

  foo();

  while (true) { // expected-warning {{jump enters an iteration statement; convergence is implementation-defined}}
                 // expected-note@#GOTO11 {{jump from this goto statement affects convergence}}
    foo();
    if (cond > 1) {
      do { // expected-warning {{jump enters an iteration statement; convergence is implementation-defined}}
           // expected-note@#GOTO11 {{jump from this goto statement affects convergence}}
        i++;
        foo();
      jumptarget:
        i++;
      } while (true);
      foo();
    }
  }

  return i;
}

__device__ int jump_into_unreachable_nest(int cond) {
  int i = 0;

  foo();

  goto jumptarget; // #GOTO43

  while (true) { // expected-warning {{jump enters an iteration statement; convergence is implementation-defined}}
                 // expected-note@#GOTO43 {{jump from this goto statement affects convergence}}
    foo();
    if (cond > 1) {
      do { // expected-warning {{jump enters an iteration statement; convergence is implementation-defined}}
           // expected-note@#GOTO43 {{jump from this goto statement affects convergence}}
        i++;
        foo();
      jumptarget:
        i++;
      } while (true);
      foo();
    }
  }

  return i;
}

__device__ int backwards_jump_into_nest(int cond) {
  int i = 0;

  while (true) {// expected-warning {{jump enters an iteration statement; convergence is implementation-defined}}
                // expected-note@#GOTO47 {{jump from this goto statement affects convergence}}
    foo();
    if (cond > 1) {
      do {// expected-warning {{jump enters an iteration statement; convergence is implementation-defined}}
          // expected-note@#GOTO47 {{jump from this goto statement affects convergence}}
        i++;
        foo();
      jumptarget:
        // expected-warning@-1 {{convergence is implementation-defined due to a backwards goto}}
        // expected-note@#GOTO47 {{jump from this goto statement affects convergence}}
        i++;
      } while (true);
      foo();
    }
  }

  if (cond > 1) {
    goto jumptarget; // #GOTO47
  }

  return i;
}

__device__ int forward_all_kinds(int cond1, int cond2, int cond3, int cond4, int cond5) {
  int i = 0;

  if (cond1 < 0)
    goto jumptarget; // #GOTO57

  for (; i < cond5;) {// expected-warning {{jump enters an iteration statement; convergence is implementation-defined}}
            // expected-note@#GOTO57 {{jump from this goto statement affects convergence}}
    foo();
    if (cond2 != 0) {
      while (i < cond3) { // expected-warning {{jump enters an iteration statement; convergence is implementation-defined}}
                     // expected-note@#GOTO57 {{jump from this goto statement affects convergence}}
        foo();
        if (cond4 > 1) {
          do { // expected-warning {{jump enters an iteration statement; convergence is implementation-defined}}
               // expected-note@#GOTO57 {{jump from this goto statement affects convergence}}
            foo();
            i++;
          jumptarget:
            i++;
          } while (i < cond4);
        }
      }
    } else {
      while (i < cond3) {
        foo();
        i++;
      }
    }
  }
  return i;
}

__device__ int forever_loops(int cond1, int cond2, int cond3) {
  int i = 0;

  if (cond1 < 0)
    goto jumptarget; // #GOTO105

  for (;;) {// expected-warning {{jump enters an iteration statement; convergence is implementation-defined}}
            // expected-note@#GOTO105 {{jump from this goto statement affects convergence}}
    foo();
    while (true) { // expected-warning {{jump enters an iteration statement; convergence is implementation-defined}}
                     // expected-note@#GOTO105 {{jump from this goto statement affects convergence}}
    jumptarget:
      i++;
      foo();
    }
  }
  return i;
}

__device__ int nest_to_nest(int cond) {
  int i = 0;

  for (;;) {
    foo();
    if (cond != 0) {
      while (true) {
        foo();
        if (cond > 1) {
          goto jumptarget; // #GOTO89
        }
      }
    }
  }

  while (true) { // expected-warning {{jump enters an iteration statement; convergence is implementation-defined}}
                 // expected-note@#GOTO89 {{jump from this goto statement affects convergence}}
    foo();
    if (cond > 1) {
      do { // expected-warning {{jump enters an iteration statement; convergence is implementation-defined}}
           // expected-note@#GOTO89 {{jump from this goto statement affects convergence}}
        foo();
        i++;
      jumptarget:
        i++;
      } while (true);
    }
  }

  return i;
}

__device__ int backwards_nest_to_nest(int cond) {
  int i = 0;

  while (true) { // expected-warning {{jump enters an iteration statement; convergence is implementation-defined}}
                 // expected-note@#GOTO131 {{jump from this goto statement affects convergence}}
    foo();
    if (cond > 1) {
      do { // expected-warning {{jump enters an iteration statement; convergence is implementation-defined}}
           // expected-note@#GOTO131 {{jump from this goto statement affects convergence}}
        foo();
        i++;
      jumptarget:
        // expected-warning@-1 {{convergence is implementation-defined due to a backwards goto}}
        // expected-note@#GOTO131 {{jump from this goto statement affects convergence}}
        i++;
      } while (true);
    }
  }

  for (;;) {
    foo();
    if (cond != 0) {
      while (true) {
        foo();
        if (cond > 1) {
          goto jumptarget; // #GOTO131
        }
      }
    }
  }

  return i;
}

__device__ int backwards_and_out(int cond) {
  int i = 0;

  if (cond > 1) {
    foo();
  jumptarget:
    // expected-warning@-1 {{convergence is implementation-defined due to a backwards goto}}
    // expected-note@#GOTO154 {{jump from this goto statement affects convergence}}
    foo();
    i++;
  }

  for (;;) {
    foo();
    if (cond != 0) {
      while (true) {
        foo();
        if (cond > 1) {
          goto jumptarget; // #GOTO154
        }
      }
    }
  }

  return i;
}

__device__ int backwards_inside_loop(int cond) {
  int i = 0;
  int j = 0;
  int k = 0;
  for (;;) {
    foo();
    i++;
  jumptarget:
    // expected-warning@-1 {{convergence is implementation-defined due to a backwards goto}}
    // expected-note@#GOTO175 {{jump from this goto statement affects convergence}}
    foo();
    j++;
    k++;
    if (cond > 5)
      goto jumptarget; // #GOTO175
  }
  return i + j + k;
}

__device__ int loop_backwards_loop(int cond) {
  int i = 0;
  int j = 0;
  int k = 0;
  for (;;) {
    foo();
    i++;
  jumptarget:
    // expected-warning@-1 {{convergence is implementation-defined due to a backwards goto}}
    // expected-note@#GOTO193 {{jump from this goto statement affects convergence}}
    j++;
    while (true) {
      foo();
      k++;
      if (cond > 5)
        goto jumptarget; // #GOTO193
    }
  }
  return i + j + k;
}

__device__ int backwards_inside_label(int cond) {
  int i = 0;
  int j = 0;
  jumptarget: {
  // expected-warning@-1 {{convergence is implementation-defined due to a backwards goto}}
  // expected-note@#GOTO208 {{jump from this goto statement affects convergence}}
    foo();
    i++;
    j++;
    if (cond > 5)
      goto jumptarget; // #GOTO208
  }
  return i + j;
}

__device__ int backwards_label_inside_branch(int cond) {
  int i = 0;
  int j = 0;
  if (cond > 0) {
  jumptarget: {
  // expected-warning@-1 {{convergence is implementation-defined due to a backwards goto}}
  // expected-note@#GOTO223 {{jump from this goto statement affects convergence}}
      foo();
      i++;
      j++;
      if (cond > 5)
        goto jumptarget; // #GOTO223
    }
  } else {
    j++;
  }
  return i + j;
}

__device__ int backwards_inside_labelled_loop(int cond) {
  int i = 0;
  int j = 0;
 jumptarget: while (true) {
  // expected-warning@-1 {{convergence is implementation-defined due to a backwards goto}}
  // expected-note@#GOTO240 {{jump from this goto statement affects convergence}}
    foo();
    i++;
    j++;
    if (cond > 5)
      goto jumptarget; // #GOTO240
  }
  return i + j;
}

// Convergence is well-defined for a jump within a switch statement.
__device__ int switch_sideways(int cond) {
  int i = 0;
  switch (cond) {
  case 10:
    foo();
    i++;
    break;
  case 20:
    foo();
    i += 2;
    goto jumptarget;
    break;
  case 30: {
    foo();
    i += 3;
  jumptarget:
    foo();
    i += 4;
    break;
  }
  default:
    foo();
    break;
  }
  return i;
}

// Convergence is implementation defined for a backward jump within a switch
// statement.
__device__ int switch_backwards(int cond) {
  int i = 0;
  switch (cond) {
  case 10:
    foo();
    i++;
    break;
  case 30: {
    foo();
    i += 3;
  jumptarget:
    // expected-warning@-1 {{convergence is implementation-defined due to a backwards goto}}
    // expected-note@#GOTO286 {{jump from this goto statement affects convergence}}
    foo();
    i += 4;
    break;
  }
  case 20:
    foo();
    i += 2;
    goto jumptarget; // #GOTO286
    break;
  default:
    foo();
    break;
  }
  return i;
}

// Convergence is well-defined with fallthrough.
__device__ int switch_fallthrough(int cond) {
  int i = 0;
  switch (cond) {
  case 10:
    foo();
    i++;
    break;
  case 20:
    foo();
    i += 2;
  case 30: {
    foo();
    i += 3;
    i += 4;
    break;
  }
  default:
    foo();
    break;
  }
  return i;
}

// Convergence is well-defined for forward jumps across a conditional statement.
__device__ int forward_conditional(int cond) {
  int i = 0;

  if (cond > 0) {
    foo();
    i++;
    goto jumptarget;
  } else {
  jumptarget:
    foo();
    i++;
  }

  return i;
}

// Convergence is implementation defined for a backwards jump across a
// conditional statement.
__device__ int backwards_conditional(int cond) {
  int i = 0;

  foo();
  if (cond > 0) {
    foo();
  jumptarget:
    // expected-warning@-1 {{convergence is implementation-defined due to a backwards goto}}
    // expected-note@#GOTO341 {{jump from this goto statement affects convergence}}
    i++;
  } else {
    foo();
    i++;
    goto jumptarget; // #GOTO341
  }

  return i;
}

__device__ int duffs_device(int count) {
  int i = 0;
  int n = (count + 7) / 8;
    switch (count % 8) {
    case 0: do { i++; // #LOOP351
        foo();
    case 7:      i++;
        // expected-warning@#LOOP351 {{jump enters an iteration statement; convergence is implementation-defined}}
        // expected-note@-2 {{jump to this case statement affects convergence of loop}}
      foo();
    case 6:      i++;
        // expected-warning@#LOOP351 {{jump enters an iteration statement; convergence is implementation-defined}}
        // expected-note@-2 {{jump to this case statement affects convergence of loop}}
    case 5:      i++;
        // expected-warning@#LOOP351 {{jump enters an iteration statement; convergence is implementation-defined}}
        // expected-note@-2 {{jump to this case statement affects convergence of loop}}
    case 4:      i++;
        // expected-warning@#LOOP351 {{jump enters an iteration statement; convergence is implementation-defined}}
        // expected-note@-2 {{jump to this case statement affects convergence of loop}}
    case 3:      i++;
        // expected-warning@#LOOP351 {{jump enters an iteration statement; convergence is implementation-defined}}
        // expected-note@-2 {{jump to this case statement affects convergence of loop}}
    case 2:      i++;
        // expected-warning@#LOOP351 {{jump enters an iteration statement; convergence is implementation-defined}}
        // expected-note@-2 {{jump to this case statement affects convergence of loop}}
    case 1:      i++;
        // expected-warning@#LOOP351 {{jump enters an iteration statement; convergence is implementation-defined}}
        // expected-note@-2 {{jump to this case statement affects convergence of loop}}
            } while (--n > 0);
    }
    return i;
}
