// RUN: %check_clang_tidy %s bugprone-cast-to-struct %t -- \
// RUN:   -config="{CheckOptions: {bugprone-cast-to-struct.IgnoredCasts: 'int \*;struct S1 \*;char \*'}}"

// CHECK-MESSAGES: warning: 'IgnoredCasts' does not contain even number of string items, the last value will be ignored. [clang-tidy-config]

struct S1 {};

void test(int *p0, short *p1) {
  struct S1 *s1;

  s1 = (struct S1 *)p0; // 'int *' to 'S1 *' ignored
  s1 = (struct S1 *)p1; // CHECK-MESSAGES: warning: casting a 'short *' pointer to a 'struct S1 *'
}
