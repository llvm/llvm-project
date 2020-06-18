// RUN: %clangxx %s -g -DSHARED_LIB -shared -o %t_shared_lib.dylib
// RUN: %clangxx %s -g -USHARED_LIB -o %t_loader
// RUN: %env_tool_opts=verbosity=3 %run %t_loader %t_shared_lib.dylib > %t_loader_output.txt 2>&1
// RUN: FileCheck -input-file=%t_loader_output.txt %s
// RUN: FileCheck -check-prefix=CHECK-STACKTRACE -input-file=%t_loader_output.txt %s
// rdar://problem/61793759 and rdar://problem/62126022.
// UNSUPPORTED: lsan

#include <stdio.h>

#ifdef SHARED_LIB
#include <sanitizer/common_interface_defs.h>

extern "C" void PrintStack() {
  fprintf(stderr, "Calling __sanitizer_print_stack_trace\n");
  // CHECK-STACKTRACE: #0{{( *0x.* *in *)?}}  __sanitizer_print_stack_trace
  // CHECK-STACKTRACE: #1{{( *0x.* *in *)?}} PrintStack {{.*}}print-stack-trace-in-code-loaded-after-fork.cpp:[[@LINE+1]]
  __sanitizer_print_stack_trace();
}
#else
#include <assert.h>
#include <dlfcn.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

typedef void (*PrintStackFnPtrTy)(void);

int main(int argc, char **argv) {
  assert(argc == 2);
  pid_t pid = fork();
  if (pid != 0) {
    // Parent
    pid_t parent_pid = getpid();
    fprintf(stderr, "parent: %d\n", parent_pid);
    int status = 0;
    pid_t child = waitpid(pid, &status, /*options=*/0);
    assert(pid == child);
    bool clean_exit = WIFEXITED(status) && WEXITSTATUS(status) == 0;
    return !clean_exit;
  }
  // Child.
  pid = getpid();
  // CHECK: child: [[CHILD_PID:[0-9]+]]
  fprintf(stderr, "child: %d\n", pid);
  // We load new code into the child process that isn't loaded into the parent.
  // When we symbolize in `PrintStack` if the symbolizer is told to symbolize
  // the parent (an old bug) rather than the child then symbolization will
  // fail.
  const char *library_to_load = argv[1];
  void *handle = dlopen(library_to_load, RTLD_NOW | RTLD_LOCAL);
  assert(handle);
  PrintStackFnPtrTy PrintStackFnPtr = (PrintStackFnPtrTy)dlsym(handle, "PrintStack");
  assert(PrintStackFnPtr);
  // Check that the symbolizer is told examine the child process.
  // CHECK: Launching Symbolizer process: {{.+}}atos -p [[CHILD_PID]]
  // CHECK-STACKTRACE: #2{{( *0x.* *in *)?}} main {{.*}}print-stack-trace-in-code-loaded-after-fork.cpp:[[@LINE+1]]
  PrintStackFnPtr();
  return 0;
}

#endif
