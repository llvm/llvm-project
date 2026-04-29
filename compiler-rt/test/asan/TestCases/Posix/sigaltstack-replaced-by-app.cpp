// Regression test for UnsetAlternateSignalStack: when another component
// installs its own alternate signal stack on top of ASan's (e.g.
// llvm/lib/Support/Unix/Signals.inc CreateSigAltStack does this when it
// finds ASan's stack too small), ASan's per-thread destructor must not
// try to free the replacement. ASan does not own that buffer; calling
// UnmapOrDie on it crashes the process at thread teardown:
//
//   AsanThread::Destroy
//     -> UnsetAlternateSignalStack
//          sigaltstack(SS_DISABLE, &oldstack)  // returns *current* stack
//          UnmapOrDie(oldstack.ss_sp, oldstack.ss_size)
//            munmap fails (the pointer was not allocated by mmap, is
//            mid-VMA, or is not page aligned)
//            -> ReportMunmapFailureAndDie -> CHECK fail -> abort
//
// Reproducer:
//   * A worker thread replaces ASan's alt-stack registration with an
//     mmap'd buffer whose ss_sp is intentionally not page aligned.
//     sigaltstack() accepts misaligned pointers; munmap() does not, so
//     ASan's eventual munmap fails deterministically with EINVAL.
//     (Without the misalignment, whether the bug fires depends on VMA
//     fragmentation, which is not reliable in a self-contained test.)
//   * Returning from the worker triggers __nptl_deallocate_tsd, which
//     runs ASan's per-thread destructor and reaches the bad path.
//     main()'s thread does not go through __nptl_deallocate_tsd on
//     process exit, which is why this only manifests with a spawned
//     thread.
//
// LeakSanitizer doesn't track raw mmap regions, so the intentionally
// leaked alt-stack mapping does not require detect_leaks=0.
//
// RUN: %clangxx_asan -O0 %s -pthread -o %t && %run %t

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

static void *worker(void *) {
  size_t pagesize = sysconf(_SC_PAGESIZE);
  size_t mapped_size = 32 * pagesize;
  char *region =
      static_cast<char *>(mmap(nullptr, mapped_size, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (region == MAP_FAILED) {
    perror("mmap");
    abort();
  }

  // ss_sp deliberately not page aligned: see file header.
  stack_t new_stack = {};
  new_stack.ss_sp = region + 1;
  new_stack.ss_size = mapped_size - 1;
  new_stack.ss_flags = 0;
  if (sigaltstack(&new_stack, nullptr) != 0) {
    perror("sigaltstack");
    abort();
  }

  // Returning here drives ASan's per-thread destructor, which is where
  // the bug used to abort.
  return nullptr;
}

int main() {
  pthread_t t;
  if (pthread_create(&t, nullptr, worker, nullptr) != 0) {
    perror("pthread_create");
    return 1;
  }
  if (pthread_join(t, nullptr) != 0) {
    perror("pthread_join");
    return 1;
  }
  printf("OK\n");
  return 0;
}
