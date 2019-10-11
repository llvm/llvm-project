// RUN: %check_clang_tidy %s misc-non-copyable-objects %t

namespace std {
typedef struct FILE {} FILE;
}
using namespace std;

// CHECK-MESSAGES: :[[@LINE+1]]:18: warning: 'f' declared as type 'FILE', which is unsafe to copy; did you mean 'FILE *'? [misc-non-copyable-objects]
void g(std::FILE f);

struct S {
  // CHECK-MESSAGES: :[[@LINE+1]]:10: warning: 'f' declared as type 'FILE', which is unsafe to copy; did you mean 'FILE *'?
  ::FILE f;
};

void func(FILE *f) {
  // CHECK-MESSAGES: :[[@LINE+1]]:13: warning: 'f1' declared as type 'FILE', which is unsafe to copy; did you mean 'FILE *'?
  std::FILE f1; // match
  // CHECK-MESSAGES: :[[@LINE+2]]:10: warning: 'f2' declared as type 'FILE', which is unsafe to copy; did you mean 'FILE *'?
  // CHECK-MESSAGES: :[[@LINE+1]]:15: warning: expression has opaque data structure type 'FILE'; type should only be used as a pointer and not dereferenced
  ::FILE f2 = *f; // match, match
  // CHECK-MESSAGES: :[[@LINE+1]]:15: warning: 'f3' declared as type 'FILE', which is unsafe to copy; did you mean 'FILE *'?
  struct FILE f3; // match
  // CHECK-MESSAGES: :[[@LINE+1]]:16: warning: expression has opaque data structure type 'FILE'; type should only be used as a pointer and not dereferenced
  (void)sizeof(*f); // match
}
