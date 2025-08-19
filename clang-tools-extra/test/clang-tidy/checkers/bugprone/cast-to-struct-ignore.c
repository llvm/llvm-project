// RUN: %check_clang_tidy %s bugprone-cast-to-struct %t -- \
// RUN:   -config="{CheckOptions: {bugprone-cast-to-struct.IgnoredCasts: '^char$;^struct S1$;^int$;Other'}}"

struct S1 {
  int a;
};

struct S2 {
  char a;
};

struct SomeOtherS {
  int a;
  int b;
};

void test1(char *p1, int *p2) {
  struct S1 *s1;
  s1 = (struct S1 *)p1;
  struct S2 *s2;
  s2 = (struct S2 *)p1;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: casting a 'char *' pointer to a 'struct S2 *'
  s2 = (struct S2 *)p2;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: casting a 'int *' pointer to a 'struct S2 *'
  struct SomeOtherS *s3;
  s3 = (struct SomeOtherS *)p2;
  s3 = (struct SomeOtherS *)p1;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: casting a 'char *' pointer to a 'struct SomeOtherS *'
}

struct S2 *test_void_is_always_ignored(void *p) {
  return (struct S2 *)p;
}
