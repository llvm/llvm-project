#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

#include "client/linux/handler/exception_handler.h"

static bool dumpCallback(const google_breakpad::MinidumpDescriptor& descriptor,
void* context, bool succeeded) {
    printf("Dump path: %s\n", descriptor.path());
    return succeeded;
}

int global = 42;

int
bar(int x, google_breakpad::ExceptionHandler &eh)
{
    eh.WriteMinidump();
    int y = 4*x + global;
    return y;
}

int
foo(int x, google_breakpad::ExceptionHandler &eh)
{
    int y = 2*bar(3*x, eh);
      return y;
}


int main(int argc, char* argv[]) {
    google_breakpad::MinidumpDescriptor descriptor("/tmp");
    google_breakpad::ExceptionHandler eh(descriptor, NULL, dumpCallback, NULL, true, -1);
    foo(1, eh);
    return 0;
}
