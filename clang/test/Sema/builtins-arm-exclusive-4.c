// RUN: %clang_cc1 -triple armv6 -fsyntax-only -verify %s

// Armv6 (apart from Armv6-M) provides 4-byte exclusive accesses, but not any
// other size. So only the calls with a pointer to a 32-bit type are accepted.

int test_ldrex(char *addr) {
  int sum = 0;
  sum += __builtin_arm_ldrex(addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 4 byte type}}
  sum += __builtin_arm_ldrex((short *)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 4 byte type}}
  sum += __builtin_arm_ldrex((int *)addr);
  sum += __builtin_arm_ldrex((long long *)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 4 byte type}}
  return sum;
}

int test_strex(char *addr) {
  int res = 0;
  res |= __builtin_arm_strex(4, addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 4 byte type}}
  res |= __builtin_arm_strex(42, (short *)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 4 byte type}}
  res |= __builtin_arm_strex(42, (int *)addr);
  res |= __builtin_arm_strex(42, (long long *)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 4 byte type}}
  return res;
}
