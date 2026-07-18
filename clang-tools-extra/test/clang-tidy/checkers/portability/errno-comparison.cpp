// RUN: %check_clang_tidy %s portability-errno-comparison %t

extern int *__errno_location();
#define errno (*__errno_location())

// A literal comparison written directly still fires in C++.
void direct() {
  if (errno == 5) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: comparing 'errno' against a literal is not portable [portability-errno-comparison]
}

// Comparing against a template parameter is not a literal as written, so no
// warning is issued and the instantiations below don't report it repeatedly.
template <int N>
bool cmp() {
  return errno == N;
}

bool a = cmp<5>();
bool b = cmp<7>();
