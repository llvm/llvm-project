#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include <stddef.h>
#include <sched.h>
#include <stdarg.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

// TSan-invisible barrier.
// Tests use it to establish necessary execution order in a way that does not
// interfere with tsan (does not establish synchronization between threads).
typedef unsigned long long invisible_barrier_t;

#ifdef __cplusplus
extern "C" {
#endif
void __tsan_testonly_barrier_init(invisible_barrier_t *barrier,
    unsigned count);
void __tsan_testonly_barrier_wait(invisible_barrier_t *barrier);
#ifdef __cplusplus
}
#endif

static inline void barrier_init(invisible_barrier_t *barrier, unsigned count) {
  __tsan_testonly_barrier_init(barrier, count);
}

static inline void barrier_wait(invisible_barrier_t *barrier) {
  __tsan_testonly_barrier_wait(barrier);
}

// Default instance of the barrier, but a test can declare more manually.
invisible_barrier_t barrier;

void print_address(const char *str, int n, ...) {
  fprintf(stderr, "%s", str);
  va_list ap;
  va_start(ap, n);
  while (n--) {
    void *p = va_arg(ap, void *);
#if defined(__x86_64__) || defined(__aarch64__) || defined(__powerpc64__)
    // On FreeBSD, the %p conversion specifier works as 0x%x and thus does not
    // match to the format used in the diagnotic message.
    fprintf(stderr, "0x%012lx ", (unsigned long) p);
#elif defined(__mips64)
    fprintf(stderr, "0x%010lx ", (unsigned long) p);
#endif
  }
  fprintf(stderr, "\n");
}

#ifdef __APPLE__
unsigned long long monotonic_clock_ns() {
  static mach_timebase_info_data_t timebase_info;
  if (timebase_info.denom == 0) mach_timebase_info(&timebase_info);
  return (mach_absolute_time() * timebase_info.numer) / timebase_info.denom;
}
#else
unsigned long long monotonic_clock_ns() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (unsigned long long)t.tv_sec * 1000000000ull + t.tv_nsec;
}
#endif

//The const kPCInc must be in sync with StackTrace::GetPreviousInstructionPc
#if defined(__powerpc64__)
// PCs are always 4 byte aligned.
const int kPCInc = 4;
#elif defined(__sparc__) || defined(__mips__)
const int kPCInc = 8;
#else
const int kPCInc = 1;
#endif

#ifdef __cplusplus
extern "C" {
#endif

void AnnotateRWLockCreate(const char *f, int l, void *m);
void AnnotateRWLockCreateStatic(const char *f, int l, void *m);
void AnnotateRWLockDestroy(const char *f, int l, void *m);
void AnnotateRWLockAcquired(const char *f, int l, void *m, long is_w);
void AnnotateRWLockReleased(const char *f, int l, void *m, long is_w);

#ifdef __cplusplus
}
#endif

#define ANNOTATE_RWLOCK_CREATE(m) \
    AnnotateRWLockCreate(__FILE__, __LINE__, m)
#define ANNOTATE_RWLOCK_CREATE_STATIC(m) \
    AnnotateRWLockCreateStatic(__FILE__, __LINE__, m)
#define ANNOTATE_RWLOCK_DESTROY(m) \
    AnnotateRWLockDestroy(__FILE__, __LINE__, m)
#define ANNOTATE_RWLOCK_ACQUIRED(m, is_w) \
    AnnotateRWLockAcquired(__FILE__, __LINE__, m, is_w)
#define ANNOTATE_RWLOCK_RELEASED(m, is_w) \
    AnnotateRWLockReleased(__FILE__, __LINE__, m, is_w)
