// RUN: %clangxx -w -fsanitize=bool %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// __ubsan_on_report is not defined as weak. Redefining it here isn't supported
// on Windows.
//
// UNSUPPORTED: win32
// Linkage issue
// XFAIL: openbsd

#include <cstdio>

extern "C" {
void __ubsan_get_current_report_data(const char **OutIssueKind,
                                     const char **OutMessage,
                                     const char **OutFilename,
                                     unsigned *OutLine, unsigned *OutCol,
                                     char **OutMemoryAddr);

// Override the definition of __ubsan_on_report from the runtime, just for
// testing purposes.
void __ubsan_on_report(void) {
  const char *IssueKind, *Message, *Filename;
  unsigned Line, Col;
  char *Addr;
  __ubsan_get_current_report_data(&IssueKind, &Message, &Filename, &Line, &Col,
                                  &Addr);

  printf("Issue: %s\n", IssueKind);
  printf("Location: %s:%u:%u\n", Filename, Line, Col);
  printf("Message: %s\n", Message);

  (void)Addr;
}
}

int main() {
  char C = 3;
  bool B = *(bool *)&C;
  // CHECK: Issue: invalid-bool-load
  // CHECK-NEXT: Location: {{.*}}monitor.cpp:[[@LINE-2]]:12
  // CHECK-NEXT: Message: Load of value 3, which is not a valid value for type 'bool'
  return 0;
}
