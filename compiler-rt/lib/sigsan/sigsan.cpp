#include "interception/interception.h" // for INTERCEPTOR, INTERCEPT_FUNCTION, REAL
#include "sanitizer_common/sanitizer_common.h"           // for Printf, Die
#include "sanitizer_common/sanitizer_report_decorator.h" // for SanitizerCommonDecorator
#include "sanitizer_common/sanitizer_stacktrace.h" // for GET_CURRENT_PC_BP_SP, BufferedStackTrace
// TODO: figure out the include for Report

#include <cstdarg>  // for va_list, va_start, va_end
#include <cstdint>  // for uintptr_t
#include <errno.h>  // for errno
#include <signal.h> // for siginfo_t, NSIG, struct sigaction, SIG_IGN, SIG_DFL
#include <stdio.h>  // for FILE

using namespace __sanitizer;

using signal_handler = void (*)(int);
using extended_signal_handler = void (*)(int, siginfo_t *, void *);

thread_local unsigned int __sigsan_signal_depth = 0;

// TODO: handle sigvec and the other deprecated sighandler installation
// functions

// TODO: Make these atomic so concurrent calls to signal for the same signum
// don't cause problems
static uintptr_t __sigsan_handlers[NSIG];

[[noreturn]] void __sigsan_print_backtrace_and_die() {
  BufferedStackTrace stack;
  GET_CURRENT_PC_BP_SP;
  (void)sp;
  stack.Unwind(pc, bp, nullptr, false);
  stack.Print();
  Die();
}

[[noreturn]] void
__sigsan_die_from_unsafe_function_call(char const *func_name) {
  __sigsan_signal_depth =
      0; /* To avoid having to make this function async-signal-safe :) */
  SanitizerCommonDecorator d;
  Printf("%s", d.Warning());
  Report("ERROR: SignalSanitizer: async-signal-unsafe function %s called from "
         "a signal handler.\n",
         func_name);
  Printf("%s", d.Default());
  __sigsan_print_backtrace_and_die();
}

[[noreturn]] void __sigsan_die_from_modified_errno() {
  __sigsan_signal_depth =
      0; /* To avoid having to make this function async-signal-safe :) */
  SanitizerCommonDecorator d;
  Printf("%s", d.Warning());
  Report("ERROR: SignalSanitizer: errno modified from a signal handler.\n");
  Printf("%s", d.Default());
  __sigsan_print_backtrace_and_die();
}

void __sigsan_handler(int signum) {
  __sigsan_signal_depth++;
  int saved_errno = errno;
  ((signal_handler)(__sigsan_handlers[signum]))(signum);
  if (errno != saved_errno) {
    __sigsan_die_from_modified_errno();
  }
  __sigsan_signal_depth--;
}

void __sigsan_extended_handler(int signum, siginfo_t *si, void *arg) {
  __sigsan_signal_depth++;
  int saved_errno = errno;
  ((extended_signal_handler)(__sigsan_handlers[signum]))(signum, si, arg);
  if (errno != saved_errno) {
    __sigsan_die_from_modified_errno();
  }
  __sigsan_signal_depth--;
}

INTERCEPTOR(signal_handler, signal, int signum, signal_handler handler) {
  // Adapted from llvm-project/libc/src/signal/linux/signal.cpp
  struct sigaction action, old;
  action.sa_handler = handler;
  action.sa_flags = SA_RESTART;
  return sigaction(signum, &action, &old) < 0 ? SIG_ERR : old.sa_handler;
}

INTERCEPTOR(int, sigaction, int sig, struct sigaction const *__restrict act,
            struct sigaction *oldact) {
  auto old_handler = __sigsan_handlers[sig];

  int result;
  if (!act || act->sa_handler == SIG_IGN || act->sa_handler == SIG_DFL) {
    result = REAL(sigaction)(sig, act, oldact);
  } else {
    // Pass in act, but replace the sa_handler with our middleman
    struct sigaction act_copy = *act;
    act_copy.sa_handler =
        act_copy.sa_flags & SA_SIGINFO
            ? (signal_handler)(uintptr_t)__sigsan_extended_handler
            : __sigsan_handler;
    result = REAL(sigaction)(sig, &act_copy, oldact);
  }

  if (result == 0) {
    if (act) {
      // TODO: Fix race condition.
      // (sig gets delievered right here, causing old sighandler to be called)
      __sigsan_handlers[sig] = (uintptr_t)act->sa_handler;
    }

    if (oldact) {
      // TODO: figure out if oldact gets written to even when result != 0

      // Stick in the handler from __sigsan_handlers, so the caller isn't aware
      // of our trickery :)
      oldact->sa_handler = (signal_handler)old_handler;
    }
  }

  return result;
}

void __sanitizer::BufferedStackTrace::UnwindImpl(uptr pc, uptr bp,
                                                 void *context,
                                                 bool request_fast,
                                                 u32 max_depth) {
  uptr top = 0;
  uptr bottom = 0;
  GetThreadStackTopAndBottom(false, &top, &bottom);
  Unwind(max_depth, pc, bp, context, top, bottom, request_fast);
}

#define SIGSAN_INTERCEPTOR(ret_type, func, args, ...)                          \
  INTERCEPTOR(ret_type, func, ##__VA_ARGS__) {                                 \
    if (__sigsan_signal_depth > 0) {                                           \
      __sigsan_die_from_unsafe_function_call(#func);                           \
    }                                                                          \
    return REAL(func) args;                                                    \
  }

// malloc
SIGSAN_INTERCEPTOR(void *, malloc, (size), size_t size)
SIGSAN_INTERCEPTOR(void *, calloc, (n, size), size_t n, size_t size)
SIGSAN_INTERCEPTOR(void *, realloc, (p, size), void *p, size_t size)
SIGSAN_INTERCEPTOR(void *, reallocarray, (p, n, size), void *p, size_t n,
                   size_t size)
SIGSAN_INTERCEPTOR(void, free, (p), void *p)

// stdio
SIGSAN_INTERCEPTOR(int, fputc, (c, stream), int c, FILE *stream)
SIGSAN_INTERCEPTOR(int, putc, (c, stream), int c, FILE *stream)
SIGSAN_INTERCEPTOR(int, putchar, (c), int c);
SIGSAN_INTERCEPTOR(int, fputs, (s, stream), const char *s, FILE *stream)
SIGSAN_INTERCEPTOR(int, puts, (s), const char *s)
SIGSAN_INTERCEPTOR(int, fflush, (stream), FILE *stream)
SIGSAN_INTERCEPTOR(int, vprintf, (format, ap), const char *format, va_list ap)
SIGSAN_INTERCEPTOR(int, vfprintf, (stream, format, ap), FILE *stream,
                   const char *format, va_list ap)
SIGSAN_INTERCEPTOR(int, vdprintf, (fd, format, ap), int fd, const char *format,
                   va_list ap)
SIGSAN_INTERCEPTOR(int, vsprintf, (str, format, ap), char *str,
                   const char *format, va_list ap)
SIGSAN_INTERCEPTOR(int, vsnprintf, (str, size, format, ap), char *str,
                   size_t size, const char *format, va_list ap)
INTERCEPTOR(int, printf, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  auto const result = REAL(vprintf)(format, ap);
  va_end(ap);
  return result;
}
INTERCEPTOR(int, fprintf, FILE *stream, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  auto const result = REAL(vfprintf)(stream, format, ap);
  va_end(ap);
  return result;
}
INTERCEPTOR(int, dprintf, int fd, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  auto const result = REAL(vdprintf)(fd, format, ap);
  va_end(ap);
  return result;
}
INTERCEPTOR(int, sprintf, char *str, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  auto const result = REAL(vsprintf)(str, format, ap);
  va_end(ap);
  return result;
}
INTERCEPTOR(int, snprintf, char *str, size_t size, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  auto const result = REAL(vsnprintf)(str, size, format, ap);
  va_end(ap);
  return result;
}
SIGSAN_INTERCEPTOR(FILE *, fopen, (path, mode), const char *path,
                   const char *mode)
SIGSAN_INTERCEPTOR(FILE *, fdopen, (fd, mode), int fd, const char *mode)
SIGSAN_INTERCEPTOR(FILE *, freopen, (path, mode, stream), const char *path,
                   const char *mode, FILE *stream)

// syslog
SIGSAN_INTERCEPTOR(void, openlog, (ident, option, facility), const char *ident,
                   int option, int facility)
SIGSAN_INTERCEPTOR(void, closelog, ())
SIGSAN_INTERCEPTOR(void, vsyslog, (priority, format, ap), int priority,
                   const char *format, va_list ap)
INTERCEPTOR(void, syslog, int priority, const char *format, ...) {
  va_list ap;
  va_start(ap, format);
  REAL(vsyslog)(priority, format, ap);
  va_end(ap);
}

SIGSAN_INTERCEPTOR(int, pthread_mutex_init, (mutex, mutexattr), pthread_mutex_t *mutex, const pthread_mutexattr_t *mutexattr)
SIGSAN_INTERCEPTOR(int, pthread_mutex_lock, (mutex), pthread_mutex_t *mutex)
SIGSAN_INTERCEPTOR(int, pthread_mutex_trylock, (mutex), pthread_mutex_t *mutex)
SIGSAN_INTERCEPTOR(int, pthread_mutex_unlock, (mutex), pthread_mutex_t *mutex)
SIGSAN_INTERCEPTOR(int, pthread_mutex_destroy, (mutex), pthread_mutex_t *mutex)

__attribute__((constructor)) void __sigsan_init() {
  SetCommonFlagsDefaults();
  InitializeCommonFlags();

  INTERCEPT_FUNCTION(signal);
  INTERCEPT_FUNCTION(sigaction);

  // malloc
  INTERCEPT_FUNCTION(malloc);
  INTERCEPT_FUNCTION(calloc);
  INTERCEPT_FUNCTION(realloc);
  INTERCEPT_FUNCTION(reallocarray);
  INTERCEPT_FUNCTION(free);

  // stdio
  INTERCEPT_FUNCTION(fputc);
  INTERCEPT_FUNCTION(putc);
  INTERCEPT_FUNCTION(putchar);
  INTERCEPT_FUNCTION(fputs);
  INTERCEPT_FUNCTION(puts);
  INTERCEPT_FUNCTION(fflush);
  INTERCEPT_FUNCTION(vprintf);
  INTERCEPT_FUNCTION(vfprintf);
  INTERCEPT_FUNCTION(vdprintf);
  INTERCEPT_FUNCTION(vsprintf);
  INTERCEPT_FUNCTION(vsnprintf);
  INTERCEPT_FUNCTION(printf);
  INTERCEPT_FUNCTION(fprintf);
  INTERCEPT_FUNCTION(dprintf);
  INTERCEPT_FUNCTION(sprintf);
  INTERCEPT_FUNCTION(snprintf);
  INTERCEPT_FUNCTION(fopen);
  INTERCEPT_FUNCTION(fdopen);
  INTERCEPT_FUNCTION(freopen);

  // syslog
  INTERCEPT_FUNCTION(openlog);
  INTERCEPT_FUNCTION(closelog);
  INTERCEPT_FUNCTION(vsyslog);
  INTERCEPT_FUNCTION(syslog);

  // pthreads
  INTERCEPT_FUNCTION(pthread_mutex_init);
  INTERCEPT_FUNCTION(pthread_mutex_lock);
  INTERCEPT_FUNCTION(pthread_mutex_trylock);
  INTERCEPT_FUNCTION(pthread_mutex_unlock);
  INTERCEPT_FUNCTION(pthread_mutex_destroy);

  // TODO: Fix race conditions.
  // (signal/sigaction called while this loop is running)
  for (int i = 0; i < NSIG; i++) {
    struct sigaction existing_sa;
    if (REAL(sigaction)(i, NULL, &existing_sa) == 0) {
      auto const existing_handler = existing_sa.sa_handler;
      if (existing_handler != SIG_IGN && existing_handler != SIG_DFL) {
        __sigsan_handlers[i] = (uintptr_t)existing_handler;
        if (existing_sa.sa_flags & SA_SIGINFO) {
          existing_sa.sa_handler = __sigsan_handler;
        } else {
          existing_sa.sa_handler =
              (signal_handler)(uintptr_t)__sigsan_extended_handler;
        }
      }
    }
  }
}
