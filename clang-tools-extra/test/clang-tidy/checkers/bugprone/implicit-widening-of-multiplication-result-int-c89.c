// RUN: %check_clang_tidy -std=c89 %s bugprone-implicit-widening-of-multiplication-result %t -- -- -target x86_64-unknown-unknown -x c

typedef long long int64_t;

int64_t t0(void) {
  return 512 * 1024;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: performing an implicit widening conversion to type 'int64_t' (aka 'long long') of a multiplication performed in type 'int' [bugprone-implicit-widening-of-multiplication-result]
  // CHECK-MESSAGES: :[[@LINE-2]]:10: note: make conversion explicit to silence this warning
  // CHECK-MESSAGES: :[[@LINE-3]]:10: note: perform multiplication in a wider type
  // CHECK-MESSAGES:                (int64_t)
}
