// RUN: %check_clang_tidy %s misc-const-correctness %t \
// RUN: -config='{CheckOptions: {misc-const-correctness.WarnPointersAsValues: true}}'
// RUN: %check_clang_tidy %s misc-const-correctness %t \
// RUN: -config='{CheckOptions: {misc-const-correctness.WarnPointersAsValues: true}}' \
// RUN: -- -DARRAY_SYNTAX

namespace ns {
int main(int argc, char **argv) {
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: pointee of variable 'argv' of type 'char **' can be declared 'const'
  // CHECK-MESSAGES: :[[@LINE-2]]:20: warning: variable 'argv' of type 'char **' can be declared 'const'
  return 0;
}
} // namespace ns

#ifdef ARRAY_SYNTAX
int main(int argc, char *argv[], char *envp[]) {
#else
int main(int argc, char **argv, char **envp) {
#endif
  int n = argc;
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: variable 'n' of type 'int' can be declared 'const'
  return n;
}
