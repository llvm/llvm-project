// RUN: %clangxx -w -fsanitize=bool -fno-sanitize-memory-param-retval %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// __ubsan_on_report is not defined as weak. Redefining it here isn't supported
// on Windows.
//
// UNSUPPORTED: windows-msvc
// Linkage issue
// XFAIL: openbsd

#include <cstdio>

// Override __ubsan_on_report() from the runtime, just for testing purposes.
// Required for dyld macOS 12.0+
#if (__APPLE__)
__attribute__((weak))
#endif
extern "C" void
__ubsan_on_report(void) {
  void __ubsan_get_current_report_data(
      const char **OutIssueKind, const char **OutMessage,
      const char **OutFilename, unsigned *OutLine, unsigned *OutCol,
      char **OutMemoryAddr);
  const char *IssueKind, *Message, *Filename;
  unsigned Line, Col;
  char *Addr;

  __ubsan_get_current_report_data(&IssueKind, &Message, &Filename, &Line, &Col,
                                  &Addr);

  printf("Issue: %s\n", IssueKind);
  printf("Location: %s:%u:%u\n", Filename, Line, Col);
  printf("Message: %s\n", Message);
  fflush(stdout);

  (void)Addr;
}

int main() {
  char C = 3;
  bool B = *(bool *)&C;
  // CHECK: Issue: invalid-bool-load
  // CHECK-NEXT: Location: {{.*}}monitor.cpp:[[@LINE-2]]:12
  // CHECK-NEXT: Message: Load of value 3, which is not a valid value for type 'bool'
  return 0;
}
