#ifndef TREC_INTERCEPTORS_H
#define TREC_INTERCEPTORS_H

#include "sanitizer_common/sanitizer_stacktrace.h"
#include "trec_rtl.h"

namespace __trec {

class ScopedInterceptor {
 public:
  ScopedInterceptor(ThreadState *thr, const char *fname, uptr pc);
  ~ScopedInterceptor();

 private:
  ThreadState *const thr_;
  const uptr pc_;
  bool should_record;
  char func_name[256];
};

}  // namespace __trec

#define SCOPED_INTERCEPTOR_RAW(func, ...)                    \
  cur_thread_init();                                         \
  ThreadState *thr = cur_thread();                           \
  const uptr caller_pc =                                     \
      StackTrace::GetPreviousInstructionPc(GET_CALLER_PC()); \
  ScopedInterceptor si(thr, #func, caller_pc);               \
  const uptr pc = caller_pc
   // StackTrace::GetCurrentPc();
/**/

#define SCOPED_TREC_INTERCEPTOR(func, ...)                           \
  SCOPED_INTERCEPTOR_RAW(func, __VA_ARGS__);                         \
  if (REAL(func) == 0) {                                             \
    Report("FATAL: TraceRecorder: failed to intercept %s\n", #func); \
    Die();                                                           \
  }                                                                  \
  if (!thr->is_inited)                                               \
    return REAL(func)(__VA_ARGS__);                                  \
  /**/

#define TREC_INTERCEPTOR(ret, func, ...) INTERCEPTOR(ret, func, __VA_ARGS__)

#if SANITIZER_NETBSD
#define TREC_INTERCEPTOR_NETBSD_ALIAS(ret, func, ...) \
  TREC_INTERCEPTOR(ret, __libc_##func, __VA_ARGS__)   \
  ALIAS(WRAPPER_NAME(pthread_##func));
#define TREC_INTERCEPTOR_NETBSD_ALIAS_THR(ret, func, ...) \
  TREC_INTERCEPTOR(ret, __libc_thr_##func, __VA_ARGS__)   \
  ALIAS(WRAPPER_NAME(pthread_##func));
#define TREC_INTERCEPTOR_NETBSD_ALIAS_THR2(ret, func, func2, ...) \
  TREC_INTERCEPTOR(ret, __libc_thr_##func, __VA_ARGS__)           \
  ALIAS(WRAPPER_NAME(pthread_##func2));
#else
#define TREC_INTERCEPTOR_NETBSD_ALIAS(ret, func, ...)
#define TREC_INTERCEPTOR_NETBSD_ALIAS_THR(ret, func, ...)
#define TREC_INTERCEPTOR_NETBSD_ALIAS_THR2(ret, func, func2, ...)
#endif

#endif  // TREC_INTERCEPTORS_H
