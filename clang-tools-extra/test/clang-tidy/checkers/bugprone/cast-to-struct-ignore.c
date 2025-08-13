// RUN: %check_clang_tidy -check-suffixes=FUNC %s bugprone-cast-to-struct %t -- \
// RUN:   -config="{CheckOptions: {bugprone-cast-to-struct.IgnoredFunctions: 'ignored_f$'}}"
// RUN: %check_clang_tidy -check-suffixes=FROM-TY %s bugprone-cast-to-struct %t -- \
// RUN:   -config="{CheckOptions: {bugprone-cast-to-struct.IgnoredFromTypes: 'int'}}"
// RUN: %check_clang_tidy -check-suffixes=TO-TY %s bugprone-cast-to-struct %t -- \
// RUN:   -config="{CheckOptions: {bugprone-cast-to-struct.IgnoredToTypes: 'IgnoredType'}}"

struct IgnoredType {
  int a;
};

struct OtherType {
  int a;
  int b;
};

void ignored_f(char *p) {
  struct OtherType *p1;
  p1 = (struct OtherType *)p;
  // CHECK-MESSAGES-FROM-TY: :[[@LINE-1]]:8: warning: casting a 'char *' pointer to a 'struct OtherType *' pointer and accessing a field can lead to memory access errors or data corruption
  // CHECK-MESSAGES-TO-TY: :[[@LINE-2]]:8: warning: casting a 'char *' pointer to a 'struct OtherType *' pointer and accessing a field can lead to memory access errors or data corruption
}

void ignored_from_type(int *p) {
  struct OtherType *p1;
  p1 = (struct OtherType *)p;
  // CHECK-MESSAGES-FUNC: :[[@LINE-1]]:8: warning: casting a 'int *' pointer to a 'struct OtherType *' pointer and accessing a field can lead to memory access errors or data corruption
  // CHECK-MESSAGES-TO-TY: :[[@LINE-2]]:8: warning: casting a 'int *' pointer to a 'struct OtherType *' pointer and accessing a field can lead to memory access errors or data corruption
}

void ignored_to_type(char *p) {
  struct IgnoredType *p1;
  p1 = (struct IgnoredType *)p;
  // CHECK-MESSAGES-FUNC: :[[@LINE-1]]:8: warning: casting a 'char *' pointer to a 'struct IgnoredType *' pointer and accessing a field can lead to memory access errors or data corruption
  // CHECK-MESSAGES-FROM-TY: :[[@LINE-2]]:8: warning: casting a 'char *' pointer to a 'struct IgnoredType *' pointer and accessing a field can lead to memory access errors or data corruption
}

struct OtherType *test_void_is_always_ignored(void *p) {
  return (struct OtherType *)p;
}
