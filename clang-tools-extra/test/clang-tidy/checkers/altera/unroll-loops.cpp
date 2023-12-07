// RUN: %check_clang_tidy %s altera-unroll-loops %t -- -config="{CheckOptions: {altera-unroll-loops.MaxLoopIterations: 50}}" -header-filter=.*
// RUN: %check_clang_tidy -check-suffix=MULT %s altera-unroll-loops %t -- -config="{CheckOptions: {altera-unroll-loops.MaxLoopIterations: 5}}" -header-filter=.* "--" -DMULT

#ifdef MULT
// For loops with *= and /= increments.
void for_loop_mult_div_increments(int *A) {
// *=
#pragma unroll
  for (int i = 2; i <= 32; i *= 2)
    A[i]++; // OK

#pragma unroll
  for (int i = 2; i <= 64; i *= 2)
    // CHECK-MESSAGES-MULT: :[[@LINE-1]]:3: warning: loop likely has a large number of iterations and thus cannot be fully unrolled; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    A[i]++; // Not OK

// /=
#pragma unroll
  for (int i = 32; i >= 2; i /= 2)
    A[i]++; // OK

#pragma unroll
  for (int i = 64; i >= 2; i /= 2)
    // CHECK-MESSAGES-MULT: :[[@LINE-1]]:3: warning: loop likely has a large number of iterations and thus cannot be fully unrolled; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    A[i]++; // Not OK
}
#else
// Cannot determine loop bounds for while loops.
void while_loops(int *A) {
  // Recommend unrolling loops that aren't already unrolled.
  int j = 0;
  while (j < 2000) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: kernel performance could be improved by unrolling this loop with a '#pragma unroll' directive [altera-unroll-loops]
    A[1] += j;
    j++;
  }

  do {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: kernel performance could be improved by unrolling this loop with a '#pragma unroll' directive [altera-unroll-loops]
    A[2] += j;
    j++;
  } while (j < 2000);

// If a while loop is fully unrolled, add a note recommending partial
// unrolling.
#pragma unroll
  while (j < 2000) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: note: full unrolling requested, but loop bounds may not be known; to partially unroll this loop, use the '#pragma unroll <num>' directive
    A[j]++;
  }

#pragma unroll
  do {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: note: full unrolling requested, but loop bounds may not be known; to partially unroll this loop, use the '#pragma unroll <num>' directive
    A[j]++;
  } while (j < 2000);

// While loop is partially unrolled, no action needed.
#pragma unroll 4
  while (j < 2000) {
    A[j]++;
  }

#pragma unroll 4
  do {
    A[j]++;
  } while (j < 2000);
}

// Range-based for loops.
void cxx_for_loops(int *A, int vectorSize) {
  // Loop with known array size should be unrolled.
  int a[] = {0, 1, 2, 3, 4, 5};
  for (int k : a) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: kernel performance could be improved by unrolling this loop with a '#pragma unroll' directive [altera-unroll-loops]
    A[k]++;
  }

// Loop with known size correctly unrolled.
#pragma unroll
  for (int k : a) {
    A[k]++;
  }

  // Loop with unknown size should be partially unrolled.
  int b[vectorSize];
#pragma unroll
  for (int k : b) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: full unrolling requested, but loop bounds are not known; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    k++;
  }

// Loop with unknown size correctly unrolled.
#pragma unroll 5
  for (int k : b) {
    k++;
  }

  // Loop with large size should be partially unrolled.
  int c[51];
#pragma unroll
  for (int k : c) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: loop likely has a large number of iterations and thus cannot be fully unrolled; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    A[k]++;
  }

// Loop with large size correctly unrolled.
#pragma unroll 5
  for (int k : c) {
    A[k]++;
  }
}

// Simple for loops.
void for_loops(int *A, int size) {
  // Recommend unrolling loops that aren't already unrolled.
  for (int i = 0; i < 2000; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: kernel performance could be improved by unrolling this loop with a '#pragma unroll' directive [altera-unroll-loops]
    A[0] += i;
  }

// Loop with known size correctly unrolled.
#pragma unroll
  for (int i = 0; i < 50; ++i) {
    A[i]++;
  }

// Loop with unknown size should be partially unrolled.
#pragma unroll
  for (int i = 0; i < size; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: full unrolling requested, but loop bounds are not known; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    A[i]++;
  }

#pragma unroll
  for (;;) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: full unrolling requested, but loop bounds are not known; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    A[0]++;
  }

  int i = 0;
#pragma unroll
  for (; i < size; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: full unrolling requested, but loop bounds are not known; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    A[i]++;
  }

#pragma unroll
  for (int i = 0;; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: full unrolling requested, but loop bounds are not known; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    A[i]++;
  }

#pragma unroll
  for (int i = 0; i < size;) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: full unrolling requested, but loop bounds are not known; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    A[i]++;
  }

#pragma unroll
  for (int i = size; i < 50; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: full unrolling requested, but loop bounds are not known; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    A[i]++;
  }

#pragma unroll
  for (int i = 0; true; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: full unrolling requested, but loop bounds are not known; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    A[i]++;
  }

#pragma unroll
  for (int i = 0; i == i; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: full unrolling requested, but loop bounds are not known; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    A[i]++;
  }

// Loop with unknown size correctly unrolled.
#pragma unroll 5
  for (int i = 0; i < size; ++i) {
    A[i]++;
  }

// Loop with large size should be partially unrolled.
#pragma unroll
  for (int i = 0; i < 51; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: loop likely has a large number of iterations and thus cannot be fully unrolled; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    A[i]++;
  }

// Loop with large size correctly unrolled.
#pragma unroll 5
  for (int i = 0; i < 51; ++i) {
    A[i]++;
  }
}

// For loops with different increments.
void for_loop_increments(int *A) {
// ++
#pragma unroll
  for (int i = 0; i < 50; ++i)
    A[i]++; // OK

#pragma unroll
  for (int i = 0; i < 51; ++i)
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: loop likely has a large number of iterations and thus cannot be fully unrolled; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    A[i]++; // Not OK

// --
#pragma unroll
  for (int i = 50; i > 0; --i)
    A[i]++; // OK

#pragma unroll
  for (int i = 51; i > 0; --i)
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: loop likely has a large number of iterations and thus cannot be fully unrolled; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    A[i]++; // Not OK

// +=
#pragma unroll
  for (int i = 0; i < 100; i += 2)
    A[i]++; // OK

#pragma unroll
  for (int i = 0; i < 101; i += 2)
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: loop likely has a large number of iterations and thus cannot be fully unrolled; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    A[i]++; // Not OK

// -=
#pragma unroll
  for (int i = 100; i > 0; i -= 2)
    A[i]++; // OK

#pragma unroll
  for (int i = 101; i > 0; i -= 2)
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: loop likely has a large number of iterations and thus cannot be fully unrolled; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    A[i]++; // Not OK
}

// Inner loops should be unrolled.
void nested_simple_loops(int *A) {
  for (int i = 0; i < 1000; ++i) {
    for (int j = 0; j < 2000; ++j) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: kernel performance could be improved by unrolling this loop with a '#pragma unroll' directive [altera-unroll-loops]
      A[0] += i + j;
    }
  }

  for (int i = 0; i < 1000; ++i) {
    int j = 0;
    while (j < 2000) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: kernel performance could be improved by unrolling this loop with a '#pragma unroll' directive [altera-unroll-loops]
      A[1] += i + j;
      j++;
    }
  }

  for (int i = 0; i < 1000; ++i) {
    int j = 0;
    do {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: kernel performance could be improved by unrolling this loop with a '#pragma unroll' directive [altera-unroll-loops]
      A[2] += i + j;
      j++;
    } while (j < 2000);
  }

  int i = 0;
  while (i < 1000) {
    for (int j = 0; j < 2000; ++j) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: kernel performance could be improved by unrolling this loop with a '#pragma unroll' directive [altera-unroll-loops]
      A[3] += i + j;
    }
    i++;
  }

  i = 0;
  while (i < 1000) {
    int j = 0;
    while (j < 2000) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: kernel performance could be improved by unrolling this loop with a '#pragma unroll' directive [altera-unroll-loops]
      A[4] += i + j;
      j++;
    }
    i++;
  }

  i = 0;
  while (i < 1000) {
    int j = 0;
    do {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: kernel performance could be improved by unrolling this loop with a '#pragma unroll' directive [altera-unroll-loops]
      A[5] += i + j;
      j++;
    } while (j < 2000);
    i++;
  }

  i = 0;
  do {
    for (int j = 0; j < 2000; ++j) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: kernel performance could be improved by unrolling this loop with a '#pragma unroll' directive [altera-unroll-loops]
      A[6] += i + j;
    }
    i++;
  } while (i < 1000);

  i = 0;
  do {
    int j = 0;
    while (j < 2000) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: kernel performance could be improved by unrolling this loop with a '#pragma unroll' directive [altera-unroll-loops]
      A[7] += i + j;
      j++;
    }
    i++;
  } while (i < 1000);

  i = 0;
  do {
    int j = 0;
    do {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: kernel performance could be improved by unrolling this loop with a '#pragma unroll' directive [altera-unroll-loops]
      A[8] += i + j;
      j++;
    } while (j < 2000);
    i++;
  } while (i < 1000);

  for (int i = 0; i < 100; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: kernel performance could be improved by unrolling this loop with a '#pragma unroll' directive [altera-unroll-loops]
    A[i]++;
  }

  i = 0;
  while (i < 100) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: kernel performance could be improved by unrolling this loop with a '#pragma unroll' directive [altera-unroll-loops]
    i++;
  }

  i = 0;
  do {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: kernel performance could be improved by unrolling this loop with a '#pragma unroll' directive [altera-unroll-loops]
    i++;
  } while (i < 100);
}

// These loops are all correctly unrolled.
void unrolled_nested_simple_loops(int *A) {
  for (int i = 0; i < 1000; ++i) {
#pragma unroll
    for (int j = 0; j < 50; ++j) {
      A[0] += i + j;
    }
  }

  for (int i = 0; i < 1000; ++i) {
    int j = 0;
#pragma unroll 5
    while (j < 50) {
      A[1] += i + j;
      j++;
    }
  }

  for (int i = 0; i < 1000; ++i) {
    int j = 0;
#pragma unroll 5
    do {
      A[2] += i + j;
      j++;
    } while (j < 50);
  }

  int i = 0;
  while (i < 1000) {
#pragma unroll
    for (int j = 0; j < 50; ++j) {
      A[3] += i + j;
    }
    i++;
  }

  i = 0;
  while (i < 1000) {
    int j = 0;
#pragma unroll 5
    while (50 > j) {
      A[4] += i + j;
      j++;
    }
    i++;
  }

  i = 0;
  while (1000 > i) {
    int j = 0;
#pragma unroll 5
    do {
      A[5] += i + j;
      j++;
    } while (j < 50);
    i++;
  }

  i = 0;
  do {
#pragma unroll
    for (int j = 0; j < 50; ++j) {
      A[6] += i + j;
    }
    i++;
  } while (i < 1000);

  i = 0;
  do {
    int j = 0;
#pragma unroll 5
    while (j < 50) {
      A[7] += i + j;
      j++;
    }
    i++;
  } while (i < 1000);

  i = 0;
  do {
    int j = 0;
#pragma unroll 5
    do {
      A[8] += i + j;
      j++;
    } while (j < 50);
    i++;
  } while (i < 1000);
}

// These inner loops are large and should be partially unrolled.
void unrolled_nested_simple_loops_large_num_iterations(int *A) {
  for (int i = 0; i < 1000; ++i) {
#pragma unroll
    for (int j = 0; j < 51; ++j) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: loop likely has a large number of iterations and thus cannot be fully unrolled; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
      A[0] += i + j;
    }
  }

  int i = 0;
  while (i < 1000) {
#pragma unroll
    for (int j = 0; j < 51; ++j) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: loop likely has a large number of iterations and thus cannot be fully unrolled; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
      A[3] += i + j;
    }
    i++;
  }

  i = 0;
  do {
#pragma unroll
    for (int j = 0; j < 51; ++j) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: loop likely has a large number of iterations and thus cannot be fully unrolled; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
      A[6] += i + j;
    }
    i++;
  } while (i < 1000);

  i = 0;
  do {
    int j = 0;
#pragma unroll
    do {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: note: full unrolling requested, but loop bounds may not be known; to partially unroll this loop, use the '#pragma unroll <num>' directive
      A[8] += i + j;
      j++;
    } while (j < 51);
    i++;
  } while (i < 1000);

  i = 0;
  int a[51];
  do {
#pragma unroll
    for (int k : a) {
      // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: loop likely has a large number of iterations and thus cannot be fully unrolled; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
      A[k]++;
    }
  } while (i < 1000);
}

// These loops have unknown bounds and should be partially unrolled.
void fully_unrolled_unknown_bounds(int vectorSize) {
  int someVector[101];

// There is no loop condition
#pragma unroll
  for (;;) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: full unrolling requested, but loop bounds are not known; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    someVector[0]++;
  }

#pragma unroll
  for (int i = 0; 1 < 5; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: full unrolling requested, but loop bounds are not known; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    someVector[i]++;
  }

// Both sides are value-dependent
#pragma unroll
  for (int i = 0; i < vectorSize; ++i) {
    // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: full unrolling requested, but loop bounds are not known; to partially unroll this loop, use the '#pragma unroll <num>' directive [altera-unroll-loops]
    someVector[i]++;
  }
}
#endif
// There are no fix-its for this check
