// RUN: %check_clang_tidy %s portability-errno-comparison %t

extern int *__errno_location(void);
#define errno (*__errno_location())
#define EINVAL 22

void positive(void) {
  if (errno == 5) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: comparing 'errno' against a literal is not portable [portability-errno-comparison]
  if (errno != 5) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: comparing 'errno' against a literal is not portable [portability-errno-comparison]
  if (errno < 10) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: comparing 'errno' against a literal is not portable [portability-errno-comparison]
  if (5 == errno) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: comparing 'errno' against a literal is not portable [portability-errno-comparison]
}

enum Err { MyErr = 5 };

#define CMP_ERR(e) ((e) == 5)

void negative(int x) {
  if (errno == 0) {}       // errno == 0 is the portable "no error" check
  if (errno != 0) {}
  if (errno == EINVAL) {}  // comparing against the named macro is the fix
  if (errno == MyErr) {}   // an enumerator is not an integer literal
  if (x == 5) {}           // not errno
  if (CMP_ERR(errno)) {}   // the comparison itself is written in a macro
}
