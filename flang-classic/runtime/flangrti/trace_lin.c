/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "flangrti_config.h"
#include <stdint.h>
#include <signal.h>
#ifdef _WIN64
#include <windows.h>
#else
#include <execinfo.h>
#include <unistd.h>
#include <limits.h>
#include <sys/types.h>
#endif
#include <stdioInterf.h>
#include "dumpregs.h"
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

/* codes and strings for signals */

struct cods {
  int code;        /* signal code */
  const char *str; /* string */
};

#define CODNULL ((struct cods *)0)

#ifndef _WIN64
static struct cods codill[] = {{ILL_ILLOPC, "illegal opcode"},
                               {ILL_ILLOPN, "illegal operand"},
                               {ILL_ILLADR, "illegal addressing mode"},
                               {ILL_ILLTRP, "illegal trap"},
                               {ILL_PRVOPC, "privileged opcode"},
                               {ILL_PRVREG, "privileged register"},
                               {ILL_COPROC, "coprocessor error"},
                               {ILL_BADSTK, "internal stack error"},
                               {0, NULL}};

static struct cods codfpe[] = {{FPE_INTDIV, "integer divide by zero"},
                               {FPE_INTOVF, "integer overflow"},
                               {FPE_FLTDIV, "floating point divide by zero"},
                               {FPE_FLTOVF, "floating point overflow"},
                               {FPE_FLTUND, "floating point underflow"},
                               {FPE_FLTRES, "floating point inexact result"},
                               {FPE_FLTINV, "floating point invalid operation"},
                               {FPE_FLTSUB, "subscript out of range"},
                               {0, NULL}};

static struct cods codsegv[] = {
    {SEGV_MAPERR, "address not mapped to object"},
    {SEGV_ACCERR, "invalid permissions for mapped object"},
    {0, NULL}};

static struct cods codbus[] = {{BUS_ADRALN, "invalid address alignment"},
                               {BUS_ADRERR, "non-existent physical address"},
                               {BUS_OBJERR, "object specific hardware error"},
                               {0, NULL}};

/* signals handled and message strings */

struct sigs {
  int sig;          /* signal value */
  struct cods *cod; /* address of optional code info */
  const char *str;  /* message string */
};

static struct sigs sigs[] = {
    {SIGINT, CODNULL, "interrupt"},
    {SIGILL, codill, "illegal instruction"},
    {SIGABRT, CODNULL, "abort"},
    {SIGFPE, codfpe, "floating point exception"},
    {SIGSEGV, codsegv, "segmentation violation"},
    {SIGTERM, CODNULL, "software termination"},
    {SIGPIPE, CODNULL, "write on a pipe with no one to read it"},
    {SIGSYS, CODNULL, "bad argument to system call"},
    {SIGHUP, CODNULL, "hangup"},
    {SIGBUS, codbus, "bus error"},
    {SIGQUIT, CODNULL, "quit"},
    {SIGTRAP, CODNULL, "trace trap"},
    {SIGIOT, CODNULL, "IOT instruction"},
    {0, CODNULL, NULL} /* end of list */
};

static FLANGRTI_GREGSET_T *regs; /* pointer to regs at signal  */

extern char **__io_get_argv();
static char **saved_argv = NULL;

/* walk the stack back */

#define MAXTRACE (32 * 1024)

void
print_back_trace_line(char *bt_str, void const *const addr)
{
  char addr2line_cmd[512], cmd_out[1024];
  char *ptr_qmark, *ptr_colon, *ptr_lparen, *ptr_rparen, *ptr_plus;
  static char buffer[4096];
  FILE *fp;
  static char exec_name[PATH_MAX]; // Includes terminating NULL
  static char *pexec_name = exec_name;

  if (NULL == saved_argv) {
    saved_argv = __io_get_argv();
  }

  if (NULL == saved_argv) {
    /*
     * Most likely a C program where __io_set_argv() was not called on startup.
     * Let's see if we can get executable's name from /proc/<PID>/cmdline.
     *
     * Here we use buffer "exec_name" to capture the executable's filename.
     * But there are issues with pathname lengths.  See:
     * https://eklitzke.org/path-max-is-tricky
     */

    FILE *f;

    snprintf(exec_name, sizeof exec_name, "/proc/%" PRIu64 "/cmdline",
             (uint64_t)getpid());
    f = fopen(exec_name, "r");
    if (NULL == f) {
      /*
       * Not sure how this could happen and if it would be better to
       * abort/exit(EXIT_FAILURE) instead of returning.
       * exit(EXIT_FAILURE);
       */
      return;
    }
    fread(exec_name, sizeof exec_name, 1, f);
    exec_name[PATH_MAX - 1] = '\0'; // In case filename+pathname > PATH_MAX
    fclose(f);
    saved_argv = &pexec_name;
  }

  sprintf(addr2line_cmd, "addr2line -e %s %p", saved_argv[0], addr);

  fp = popen(addr2line_cmd, "r");
  if (fp != NULL) {
    if (fgets(cmd_out, sizeof(cmd_out) - 1, fp) != NULL) {
      if (cmd_out[strlen(cmd_out) - 1] == '\n')
        cmd_out[strlen(cmd_out) - 1] = '\0';
      ptr_qmark = strchr(cmd_out, '?');
      ptr_colon = strchr(cmd_out, ':');
      ptr_lparen = strchr(bt_str, '(');
      ptr_rparen = strchr(bt_str, ')');
      if (ptr_lparen != NULL && ptr_lparen != NULL && ptr_lparen < ptr_rparen)
        ptr_plus = strchr(ptr_lparen, '+');
      if (ptr_qmark != NULL) {
        fprintf(__io_stderr(), "  %s\n", bt_str);
      } else if (ptr_colon != NULL && ptr_plus != NULL &&
                 ptr_plus < ptr_rparen) {
        /* replace offset by line number */
        strncpy(buffer, bt_str, ptr_plus - bt_str);
        sprintf(buffer + (ptr_plus - bt_str), "%s%s", ptr_colon, ptr_rparen);
        fprintf(__io_stderr(), "  %s\n", buffer);
      }
    } else {
      fprintf(__io_stderr(), "  %s\n", bt_str);
    }
    fclose(fp);
  } else {
    fprintf(__io_stderr(), "  %s\n", bt_str);
  }
}

void
__abort_trace(int skip)
{
  void *array[MAXTRACE];
  size_t size;
  char **strings;
  size_t i;

  if (regs != (FLANGRTI_GREGSET_T *)0) {
    dumpregs(regs);
  }

  size = backtrace(array, MAXTRACE);
  if (skip + 1 >= size) {
    fprintf(__io_stderr(), "  --- traceback not available\n");
    return;
  }
  strings = backtrace_symbols(array, size);
  if (size < 100) {
    for (i = skip + 1; i < size; i++)
      print_back_trace_line(strings[i], array[i]);
  } else {
    for (i = skip + 1; i < 40; i++)
      print_back_trace_line(strings[i], array[i]);
    fprintf(__io_stderr(), "  --- skipping traceback entries\n");
    for (i = size - 40; i < size; i++)
      print_back_trace_line(strings[i], array[i]);
  }
  free(strings);
}

/*
 * this routine is a signal handler, it prints a message and terminates the
 * process
 */

static void
__abort_sig_hand(int sig, siginfo_t *in, FLANGRTI_UCONTEXT_T *u)
{
  const char *p;
  char b[128];
  int n, m;
  struct sigaction new;
  struct sigaction old;

  new.sa_sigaction = (void (*)(int, siginfo_t *, void *))SIG_DFL;
  sigemptyset(&new.sa_mask);
  new.sa_flags = SA_SIGINFO;
  n = 0;
  while (sigs[n].sig != 0) {
    sigaction(sigs[n].sig, &new, &old);
    n++;
  }

  regs = getRegs(u);

  n = 0;
  while ((sigs[n].sig != 0) && (sigs[n].sig != sig)) {
    n++;
  }
  if (sigs[n].sig == 0) {
    sprintf(b, "signal %d", sig);
    p = b;
  } else {
    p = sigs[n].str;
    m = 0;
    if (sigs[n].cod != CODNULL) {
      while ((sigs[n].cod[m].code != 0) &&
             (sigs[n].cod[m].code != in->si_code)) {
        m++;
      }
      if (sigs[n].cod[m].code != 0) {
        sprintf(b, "%s, %s", p, sigs[n].cod[m].str);
        p = b;
      }
    }
  }
  __abort(3, p);
}

/*
 * this routine initializes the signal handlers
 */

void
__abort_sig_init(void)
{
  struct sigaction new;
  struct sigaction old;
  int n;

  new.sa_sigaction = (void (*)(int, siginfo_t *, void *))__abort_sig_hand;
  sigemptyset(&new.sa_mask);
  new.sa_flags = SA_SIGINFO;
  n = 0;
  while (sigs[n].sig != 0) {
    sigaction(sigs[n].sig, &new, &old);
    n++;
  }
}
#else
void
__abort_trace(int skip)
{
  // TODO
  return;
}

LONG WINAPI
windows_exception_handler(EXCEPTION_POINTERS *ExceptionInfo)
{
  switch (ExceptionInfo->ExceptionRecord->ExceptionCode) {
  case EXCEPTION_ACCESS_VIOLATION:
    fputs("Error: EXCEPTION_ACCESS_VIOLATION\n", stderr);
    break;
  case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
    fputs("Error: EXCEPTION_ARRAY_BOUNDS_EXCEEDED\n", stderr);
    break;
  case EXCEPTION_BREAKPOINT:
    fputs("Error: EXCEPTION_BREAKPOINT\n", stderr);
    break;
  case EXCEPTION_DATATYPE_MISALIGNMENT:
    fputs("Error: EXCEPTION_DATATYPE_MISALIGNMENT\n", stderr);
    break;
  case EXCEPTION_FLT_DENORMAL_OPERAND:
    fputs("Error: EXCEPTION_FLT_DENORMAL_OPERAND\n", stderr);
    break;
  case EXCEPTION_FLT_DIVIDE_BY_ZERO:
    fputs("Error: EXCEPTION_FLT_DIVIDE_BY_ZERO\n", stderr);
    break;
  case EXCEPTION_FLT_INEXACT_RESULT:
    fputs("Error: EXCEPTION_FLT_INEXACT_RESULT\n", stderr);
    break;
  case EXCEPTION_FLT_INVALID_OPERATION:
    fputs("Error: EXCEPTION_FLT_INVALID_OPERATION\n", stderr);
    break;
  case EXCEPTION_FLT_OVERFLOW:
    fputs("Error: EXCEPTION_FLT_OVERFLOW\n", stderr);
    break;
  case EXCEPTION_FLT_STACK_CHECK:
    fputs("Error: EXCEPTION_FLT_STACK_CHECK\n", stderr);
    break;
  case EXCEPTION_FLT_UNDERFLOW:
    fputs("Error: EXCEPTION_FLT_UNDERFLOW\n", stderr);
    break;
  case EXCEPTION_ILLEGAL_INSTRUCTION:
    fputs("Error: EXCEPTION_ILLEGAL_INSTRUCTION\n", stderr);
    break;
  case EXCEPTION_IN_PAGE_ERROR:
    fputs("Error: EXCEPTION_IN_PAGE_ERROR\n", stderr);
    break;
  case EXCEPTION_INT_DIVIDE_BY_ZERO:
    fputs("Error: EXCEPTION_INT_DIVIDE_BY_ZERO\n", stderr);
    break;
  case EXCEPTION_INT_OVERFLOW:
    fputs("Error: EXCEPTION_INT_OVERFLOW\n", stderr);
    break;
  case EXCEPTION_INVALID_DISPOSITION:
    fputs("Error: EXCEPTION_INVALID_DISPOSITION\n", stderr);
    break;
  case EXCEPTION_NONCONTINUABLE_EXCEPTION:
    fputs("Error: EXCEPTION_NONCONTINUABLE_EXCEPTION\n", stderr);
    break;
  case EXCEPTION_PRIV_INSTRUCTION:
    fputs("Error: EXCEPTION_PRIV_INSTRUCTION\n", stderr);
    break;
  case EXCEPTION_SINGLE_STEP:
    fputs("Error: EXCEPTION_SINGLE_STEP\n", stderr);
    break;
  case EXCEPTION_STACK_OVERFLOW:
    fputs("Error: EXCEPTION_STACK_OVERFLOW\n", stderr);
    break;
  default:
    fputs("Error: Unrecognized Exception\n", stderr);
    break;
  }
  fflush(stderr);
  return EXCEPTION_EXECUTE_HANDLER;
}

void
__abort_sig_init(void)
{
  SetUnhandledExceptionFilter(windows_exception_handler);
}
#endif
