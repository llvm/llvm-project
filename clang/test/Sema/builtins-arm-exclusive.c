// RUN: %clang_cc1 -triple armv7 -fsyntax-only -verify %s

// General tests of __builtin_arm_ldrex[d] and __builtin_arm_strex[d] error checking.
//
// This test is compiled for Armv7-A, which provides exclusive load/store
// instructions for 1-, 2-, 4- and 8-byte quantities. Other Arm architecture
// versions provide subsets of those, requiring different error reporting.
// Those are tested in builtins-arm-exclusive-124.c, builtins-arm-exclusive-4.c
// and builtins-arm-exclusive-none.c.

struct Simple {
  char a, b;
};

int test_ldrex(char *addr) {
  int sum = 0;
  sum += __builtin_arm_ldrex(addr);
  sum += __builtin_arm_ldrex((short *)addr);
  sum += __builtin_arm_ldrex((int *)addr);
  sum += __builtin_arm_ldrex((long long *)addr);
  sum += __builtin_arm_ldrex((float *)addr);
  sum += __builtin_arm_ldrex((double *)addr);
  sum += *__builtin_arm_ldrex((int **)addr);
  sum += __builtin_arm_ldrex((struct Simple **)addr)->a;
  sum += __builtin_arm_ldrex((volatile char *)addr);
  sum += __builtin_arm_ldrex((const volatile char *)addr);

  // In principle this might be valid, but stick to ints and floats for scalar
  // types at the moment.
  sum += __builtin_arm_ldrex((struct Simple *)addr).a; // expected-error {{address argument to atomic builtin must be a pointer to}}

  sum += __builtin_arm_ldrex((__int128 *)addr); // expected-error {{__int128 is not supported on this target}} expected-error {{address argument to load or store exclusive builtin must be a pointer to 1,2,4 or 8 byte type}}

  __builtin_arm_ldrex(); // expected-error {{too few arguments to function call}}
  __builtin_arm_ldrex(1, 2); // expected-error {{too many arguments to function call}}
  return sum;
}

int test_strex(char *addr) {
  int res = 0;
  struct Simple var = {0};
  res |= __builtin_arm_strex(4, addr);
  res |= __builtin_arm_strex(42, (short *)addr);
  res |= __builtin_arm_strex(42, (int *)addr);
  res |= __builtin_arm_strex(42, (long long *)addr);
  res |= __builtin_arm_strex(2.71828f, (float *)addr);
  res |= __builtin_arm_strex(3.14159, (double *)addr);
  res |= __builtin_arm_strex(&var, (struct Simple **)addr);

  res |= __builtin_arm_strex(42, (volatile char *)addr);
  res |= __builtin_arm_strex(42, (char *const)addr);
  res |= __builtin_arm_strex(42, (const char *)addr); // expected-warning {{passing 'const char *' to parameter of type 'volatile char *' discards qualifiers}}


  res |= __builtin_arm_strex(var, (struct Simple *)addr); // expected-error {{address argument to atomic builtin must be a pointer to}}
  res |= __builtin_arm_strex(var, (struct Simple **)addr); // expected-error {{passing 'struct Simple' to parameter of incompatible type 'struct Simple *'}}
  res |= __builtin_arm_strex(&var, (struct Simple **)addr).a; // expected-error {{is not a structure or union}}

  res |= __builtin_arm_strex(1, (__int128 *)addr); // expected-error {{__int128 is not supported on this target}} expected-error {{address argument to load or store exclusive builtin must be a pointer to 1,2,4 or 8 byte type}}

  __builtin_arm_strex(1); // expected-error {{too few arguments to function call}}
  __builtin_arm_strex(1, 2, 3); // expected-error {{too many arguments to function call}}
  return res;
}

int test_ldrexd(char *addr) {
  int sum = 0;
  sum += __builtin_arm_ldrexd(addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}
  sum += __builtin_arm_ldrexd((short *)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}
  sum += __builtin_arm_ldrexd((int *)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}
  sum += __builtin_arm_ldrexd((long long *)addr);
  sum += __builtin_arm_ldrexd((float *)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}
  sum += __builtin_arm_ldrexd((double *)addr);
  sum += *__builtin_arm_ldrexd((int **)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}
  sum += __builtin_arm_ldrexd((struct Simple **)addr)->a; // expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}
  sum += __builtin_arm_ldrexd((volatile char *)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}
  sum += __builtin_arm_ldrexd((const volatile char *)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}

  // In principle this might be valid, but stick to ints and floats for scalar
  // types at the moment.
  sum += __builtin_arm_ldrexd((struct Simple *)addr).a; // expected-error {{address argument to atomic builtin must be a pointer to}}

  sum += __builtin_arm_ldrexd((__int128 *)addr); // expected-error {{__int128 is not supported on this target}} expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}

  __builtin_arm_ldrexd(); // expected-error {{too few arguments to function call}}
  __builtin_arm_ldrexd(1, 2); // expected-error {{too many arguments to function call}}
  return sum;
}

int test_strexd(char *addr) {
  int res = 0;
  struct Simple var = {0};
  res |= __builtin_arm_strexd(4, addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}
  res |= __builtin_arm_strexd(42, (short *)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}
  res |= __builtin_arm_strexd(42, (int *)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}
  res |= __builtin_arm_strexd(42, (long long *)addr);
  res |= __builtin_arm_strexd(2.71828f, (float *)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}
  res |= __builtin_arm_strexd(3.14159, (double *)addr);
  res |= __builtin_arm_strexd(&var, (struct Simple **)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}

  res |= __builtin_arm_strexd(42, (volatile char *)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}
  res |= __builtin_arm_strexd(42, (char *const)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}
  res |= __builtin_arm_strexd(42, (const char *)addr); // expected-warning {{passing 'const char *' to parameter of type 'volatile char *' discards qualifiers}} expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}


  res |= __builtin_arm_strexd(var, (struct Simple *)addr); // expected-error {{address argument to atomic builtin must be a pointer to}}
  res |= __builtin_arm_strexd(var, (struct Simple **)addr); // expected-error {{passing 'struct Simple' to parameter of incompatible type 'struct Simple *'}} expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}
  res |= __builtin_arm_strexd(&var, (struct Simple **)addr).a; // expected-error {{is not a structure or union}} expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}

  res |= __builtin_arm_strexd(1, (__int128 *)addr); // expected-error {{__int128 is not supported on this target}} expected-error {{address argument to load or store exclusive builtin must be a pointer to 8 byte type}}

  __builtin_arm_strexd(1); // expected-error {{too few arguments to function call}}
  __builtin_arm_strexd(1, 2, 3); // expected-error {{too many arguments to function call}}
  return res;
}

int test_ldaex(char *addr) {
  int sum = 0;
  sum += __builtin_arm_ldaex(addr);
  sum += __builtin_arm_ldaex((short *)addr);
  sum += __builtin_arm_ldaex((int *)addr);
  sum += __builtin_arm_ldaex((long long *)addr);
  sum += __builtin_arm_ldaex((float *)addr);
  sum += __builtin_arm_ldaex((double *)addr);
  sum += *__builtin_arm_ldaex((int **)addr);
  sum += __builtin_arm_ldaex((struct Simple **)addr)->a;
  sum += __builtin_arm_ldaex((volatile char *)addr);
  sum += __builtin_arm_ldaex((const volatile char *)addr);

  // In principle this might be valid, but stick to ints and floats for scalar
  // types at the moment.
  sum += __builtin_arm_ldaex((struct Simple *)addr).a; // expected-error {{address argument to atomic builtin must be a pointer to}}

  sum += __builtin_arm_ldaex((__int128 *)addr); // expected-error {{__int128 is not supported on this target}} expected-error {{address argument to load or store exclusive builtin must be a pointer to 1,2,4 or 8 byte type}}

  __builtin_arm_ldaex(); // expected-error {{too few arguments to function call}}
  __builtin_arm_ldaex(1, 2); // expected-error {{too many arguments to function call}}
  return sum;
}

int test_stlex(char *addr) {
  int res = 0;
  struct Simple var = {0};
  res |= __builtin_arm_stlex(4, addr);
  res |= __builtin_arm_stlex(42, (short *)addr);
  res |= __builtin_arm_stlex(42, (int *)addr);
  res |= __builtin_arm_stlex(42, (long long *)addr);
  res |= __builtin_arm_stlex(2.71828f, (float *)addr);
  res |= __builtin_arm_stlex(3.14159, (double *)addr);
  res |= __builtin_arm_stlex(&var, (struct Simple **)addr);

  res |= __builtin_arm_stlex(42, (volatile char *)addr);
  res |= __builtin_arm_stlex(42, (char *const)addr);
  res |= __builtin_arm_stlex(42, (const char *)addr); // expected-warning {{passing 'const char *' to parameter of type 'volatile char *' discards qualifiers}}


  res |= __builtin_arm_stlex(var, (struct Simple *)addr); // expected-error {{address argument to atomic builtin must be a pointer to}}
  res |= __builtin_arm_stlex(var, (struct Simple **)addr); // expected-error {{passing 'struct Simple' to parameter of incompatible type 'struct Simple *'}}
  res |= __builtin_arm_stlex(&var, (struct Simple **)addr).a; // expected-error {{is not a structure or union}}

  res |= __builtin_arm_stlex(1, (__int128 *)addr); // expected-error {{__int128 is not supported on this target}} expected-error {{address argument to load or store exclusive builtin must be a pointer to 1,2,4 or 8 byte type}}

  __builtin_arm_stlex(1); // expected-error {{too few arguments to function call}}
  __builtin_arm_stlex(1, 2, 3); // expected-error {{too many arguments to function call}}
  return res;
}

void test_clrex(void) {
  __builtin_arm_clrex();
  __builtin_arm_clrex(1); // expected-error {{too many arguments to function call}}
}
