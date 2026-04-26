// RUN: %clang_cc1 -triple thumbv7m -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple thumbv8m.main -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple armv8.1m.main -fsyntax-only -verify %s

// All these architecture versions provide 1-, 2- or 4-byte exclusive accesses,
// but don't have the LDREXD instruction which takes two operand registers and
// performs an 8-byte exclusive access. So the calls with a pointer to long
// long are rejected.

int test_ldrex(char *addr) {
  int sum = 0;
  sum += __builtin_arm_ldrex(addr);
  sum += __builtin_arm_ldrex((short *)addr);
  sum += __builtin_arm_ldrex((int *)addr);
  sum += __builtin_arm_ldrex((long long *)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 1,2 or 4 byte type}}
  return sum;
}

int test_strex(char *addr) {
  int res = 0;
  res |= __builtin_arm_strex(4, addr);
  res |= __builtin_arm_strex(42, (short *)addr);
  res |= __builtin_arm_strex(42, (int *)addr);
  res |= __builtin_arm_strex(42, (long long *)addr); // expected-error {{address argument to load or store exclusive builtin must be a pointer to 1,2 or 4 byte type}}
  return res;
}

int test_ldrexd(char *addr) {
  int sum = 0;
  sum += __builtin_arm_ldrexd(addr); // expected-error {{eight-byte load and store exclusive builtins are not available on this architecture}}
  sum += __builtin_arm_ldrexd((short *)addr); // expected-error {{eight-byte load and store exclusive builtins are not available on this architecture}}
  sum += __builtin_arm_ldrexd((int *)addr); // expected-error {{eight-byte load and store exclusive builtins are not available on this architecture}}
  sum += __builtin_arm_ldrexd((long long *)addr); // expected-error {{eight-byte load and store exclusive builtins are not available on this architecture}}
  sum += __builtin_arm_ldrexd((unsigned long long *)addr); // expected-error {{eight-byte load and store exclusive builtins are not available on this architecture}}
  sum += __builtin_arm_ldrexd((float *)addr); // expected-error {{eight-byte load and store exclusive builtins are not available on this architecture}}
  sum += __builtin_arm_ldrexd((double *)addr); // expected-error {{eight-byte load and store exclusive builtins are not available on this architecture}}
  return sum;
}

int test_strexd(char *addr) {
  int res = 0;
  res |= __builtin_arm_strexd(4, addr); // expected-error {{eight-byte load and store exclusive builtins are not available on this architecture}}
  res |= __builtin_arm_strexd(42, (short *)addr); // expected-error {{eight-byte load and store exclusive builtins are not available on this architecture}}
  res |= __builtin_arm_strexd(42, (int *)addr); // expected-error {{eight-byte load and store exclusive builtins are not available on this architecture}}
  res |= __builtin_arm_strexd(42, (long long *)addr); // expected-error {{eight-byte load and store exclusive builtins are not available on this architecture}}
  res |= __builtin_arm_strexd(42, (unsigned long long *)addr); // expected-error {{eight-byte load and store exclusive builtins are not available on this architecture}}
  res |= __builtin_arm_strexd(2.71828f, (float *)addr); // expected-error {{eight-byte load and store exclusive builtins are not available on this architecture}}
  res |= __builtin_arm_strexd(3.14159, (double *)addr); // expected-error {{eight-byte load and store exclusive builtins are not available on this architecture}}
  return res;
}
