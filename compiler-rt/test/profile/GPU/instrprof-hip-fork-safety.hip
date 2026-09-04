// Fork safety: loading the profile-instrumented library must NOT initialize
// HSA in a process that itself never touches HIP and only runs device work in
// forked children. RCCL's unit tests follow exactly this pattern -- the parent
// deliberately keeps HIP/HSA uninitialized and launches kernels only inside
// forked children. If the profile runtime's library constructor eagerly called
// hsa_init(), the child would inherit invalid HSA state across fork() and crash
// inside HSA (e.g. hsa_amd_signal_create -> SharedSignalPool::alloc). The HSA
// drain therefore brings HSA up lazily, never from a constructor.
//
// REQUIRES: hip, amdgpu
// The eager-hsa_init fork hazard and the lazy HSA drain are Linux-only.
// UNSUPPORTED: windows
// RUN: %clang -x hip -fprofile-instr-generate -fcoverage-mapping \
// RUN:   --offload-arch=%amdgpu_arch %s -o %t -L%hip_lib_path -lamdhip64
// RUN: env LLVM_PROFILE_FILE=%t.profraw \
// RUN:   LD_LIBRARY_PATH=%hip_lib_path:$LD_LIBRARY_PATH \
// RUN:   %run %t 2>&1 | FileCheck %s
//
// The forked child must complete its kernel without crashing in HSA.
// CHECK: PASS

#include <hip/hip_runtime.h>
#include <cstdio>
#include <sys/wait.h>
#include <unistd.h>

__global__ void increment(int *p) { *p += 1; }

static int doChildWork() {
  int *d = nullptr;
  if (hipMalloc(&d, sizeof(int)) != hipSuccess)
    return 1;
  int h = 41;
  (void)hipMemcpy(d, &h, sizeof(int), hipMemcpyHostToDevice);
  increment<<<1, 1>>>(d);
  if (hipDeviceSynchronize() != hipSuccess)
    return 1;
  (void)hipMemcpy(&h, d, sizeof(int), hipMemcpyDeviceToHost);
  (void)hipFree(d);
  return h == 42 ? 0 : 1;
}

int main() {
  // The parent intentionally performs no HIP/HSA work before forking.
  pid_t pid = fork();
  if (pid < 0) {
    printf("FAIL (fork failed)\n");
    return 1;
  }
  if (pid == 0) {
    // Child runs the device work; _exit avoids flushing the parent's profile
    // handlers from the child (the RCCL test pattern).
    _exit(doChildWork());
  }

  int status = 0;
  (void)waitpid(pid, &status, 0);
  int ok = WIFEXITED(status) && WEXITSTATUS(status) == 0;
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
