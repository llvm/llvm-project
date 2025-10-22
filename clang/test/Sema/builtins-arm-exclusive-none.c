// RUN: %clang_cc1 -triple armv6m -fsyntax-only -verify %s

// Armv6-M does not support exclusive loads/stores at all, so all uses of
// __builtin_arm_ldrex and __builtin_arm_strex is forbidden.

int test_ldrex(char *addr) {
  int sum = 0;
  sum += __builtin_arm_ldrex(addr); // expected-error {{load and store exclusive builtins are not available on this architecture}}
  sum += __builtin_arm_ldrex((short *)addr); // expected-error {{load and store exclusive builtins are not available on this architecture}}
  sum += __builtin_arm_ldrex((int *)addr); // expected-error {{load and store exclusive builtins are not available on this architecture}}
  sum += __builtin_arm_ldrex((long long *)addr); // expected-error {{load and store exclusive builtins are not available on this architecture}}
  return sum;
}

int test_strex(char *addr) {
  int res = 0;
  res |= __builtin_arm_strex(4, addr); // expected-error {{load and store exclusive builtins are not available on this architecture}}
  res |= __builtin_arm_strex(42, (short *)addr); // expected-error {{load and store exclusive builtins are not available on this architecture}}
  res |= __builtin_arm_strex(42, (int *)addr); // expected-error {{load and store exclusive builtins are not available on this architecture}}
  res |= __builtin_arm_strex(42, (long long *)addr); // expected-error {{load and store exclusive builtins are not available on this architecture}}
  return res;
}
