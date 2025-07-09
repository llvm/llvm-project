// RUN: %check_clang_tidy -check-suffixes=ENABLE-IN-C %s bugprone-multi-level-implicit-pointer-conversion %t -- -config="{CheckOptions: {bugprone-multi-level-implicit-pointer-conversion.EnableInC: true}}"
// RUN: %check_clang_tidy -check-suffixes=DISABLE-IN-C %s bugprone-multi-level-implicit-pointer-conversion %t -- -config="{CheckOptions: {bugprone-multi-level-implicit-pointer-conversion.EnableInC: false}}"

void free(void*);

void test() {
  char **p;
  free(p);
  // CHECK-MESSAGES-ENABLE-IN-C: :[[@LINE-1]]:8: warning: multilevel pointer conversion from 'char **' to 'void *', please use explicit cast [bugprone-multi-level-implicit-pointer-conversion]
  free((void *)p);
}
