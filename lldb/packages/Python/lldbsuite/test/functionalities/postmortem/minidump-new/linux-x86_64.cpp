// Example source from breakpad's linux tutorial
// https://chromium.googlesource.com/breakpad/breakpad/+/master/docs/linux_starter_guide.md

#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

#include "client/linux/handler/exception_handler.h"

static bool dumpCallback(const google_breakpad::MinidumpDescriptor &descriptor,
                         void *context, bool succeeded) {
  printf("Dump path: %s\n", descriptor.path());
  return succeeded;
}

void crash() {
  volatile int *a = (int *)(NULL);
  *a = 1;
}

int main(int argc, char *argv[]) {
  google_breakpad::MinidumpDescriptor descriptor("/tmp");
  google_breakpad::ExceptionHandler eh(descriptor, NULL, dumpCallback, NULL,
                                       true, -1);
  printf("pid: %d\n", getpid());
  crash();
  return 0;
}
