// RUN: %clangxx -w -fsanitize=bool %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// __ubsan_on_report is not defined as weak. Redefining it here isn't supported
// on Windows.
//
// UNSUPPORTED: win32

#include <iostream>

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

  std::cout << "Issue: " << IssueKind << "\n"
            << "Location: " << Filename << ":" << Line << ":" << Col << "\n"
            << "Message: " << Message << std::endl;

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
