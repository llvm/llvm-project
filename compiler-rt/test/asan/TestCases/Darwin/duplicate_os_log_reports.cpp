// UNSUPPORTED: ios
// REQUIRES: darwin_log_cmd
// RUN: %clangxx_asan -fsanitize-recover=address %s -o %t
// RUN: { %env_asan_opts=halt_on_error=0,log_to_syslog=1 %run %t > %t.process_output.txt 2>&1 & } \
// RUN: ; export TEST_PID=$! ; wait ${TEST_PID}

// Check process output.
// RUN: FileCheck %s --check-prefixes CHECK,CHECK-PROC -input-file=%t.process_output.txt

// Check syslog output. We filter recent system logs based on PID to avoid
// getting the logs of previous test runs.
// RUN: log show --debug --last 5m  --predicate "processID == ${TEST_PID}" --style syslog > %t.process_syslog_output.txt
// RUN: FileCheck %s -input-file=%t.process_syslog_output.txt
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sanitizer/asan_interface.h>

const int kBufferSize = 512;
char *buffer;

// `readZero` and `readOne` exist so that we can distinguish the two
// error reports based on the symbolized stacktrace.
void readZero() {
  assert(__asan_address_is_poisoned(buffer));
  char c = buffer[0];
  printf("Read %c\n", c);
}

void readOne() {
  assert(__asan_address_is_poisoned(buffer + 1));
  char c = buffer[1];
  printf("Read %c\n", c);
}

int main() {
  buffer = static_cast<char *>(malloc(kBufferSize));
  memset(static_cast<void *>(buffer), static_cast<int>('.'), kBufferSize);
  assert(buffer);
  // Deliberately poison `buffer` so that we have a deterministic way
  // triggering two ASan reports in a row in the no halt_on_error mode (e.g. Two
  // heap-use-after free in a row might not be deterministic).
  __asan_poison_memory_region(buffer, kBufferSize);

  // This sequence of ASan reports are designed to catch an old bug in the way
  // ASan's internal syslog buffer was handled after reporting an issue.
  // Previously in the no halt_on_error mode the internal buffer wasn't cleared
  // after reporting an issue. When another issue was encountered everything
  // that was already in the buffer would be written to the syslog again
  // leading to duplicate reports in the syslog.

  // First bad access.
  // CHECK: use-after-poison
  // CHECK-NEXT: READ of size 1
  // CHECK-NEXT: #0 0x{{[0-9a-f]+}} in readZero
  // CHECK: SUMMARY: {{.*}} use-after-poison {{.*}} in readZero
  readZero();

  // Second bad access.
  // CHECK: use-after-poison
  // CHECK-NEXT: READ of size 1
  // CHECK-NEXT: #0 0x{{[0-9a-f]+}} in readOne
  // CHECK: SUMMARY: {{.*}} use-after-poison {{.*}} in readOne
  readOne();

  // CHECK-PROC: DONE
  printf("DONE\n");
  return 0;
}
