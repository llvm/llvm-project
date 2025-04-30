/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

/** \file
 * \brief FIXME
 */

#include "stdioInterf.h"
#include "fioMacros.h"

WIN_API __INT_T LINENO[];

#if defined(_WIN64)
#include <io.h> // for _write
#define write _write
#endif

/* function stack entry */

struct pent {
  const char *func; /* function name (no \0) */
  __CLEN_T funcl;   /* length of above */
  const char *file; /* file name (no \0) */
  __CLEN_T filel;   /* length of above */
  int line;   /* line number of function entry */
  int lines;  /* number of lines in function */
  int cline;  /* calling function line number */
};

#define PENTS 256           /* initial number of stack entries */
#define UNKNOWN "<unknown>" /* function/file name of dummy entry */

static struct pent *pentb;
static struct pent *pentc;
static struct pent *pente;

static int __fort_stat_mflag;
static int __fort_trac_mflag;
static int __fort_prof_mflag;
int __fort_entry_mflag;

#if defined(_WIN64)
/* pg access for dlls */
char *
__get_fort_lineno_addr(void)
{
  return (char *)ENTCOMN(LINENO, lineno);
}

#endif

/** \brief Initialization */
void
__fort_entry_init(void)
{
  pentb = (struct pent *)__fort_malloc(PENTS * sizeof(struct pent));
  pentc = pentb;
  pente = pentb + PENTS;
  pentc->func = UNKNOWN;
  pentc->funcl = sizeof(UNKNOWN);
  pentc->file = UNKNOWN;
  pentc->filel = sizeof(UNKNOWN);
  __fort_stat_mflag = __fort_stat_init();
  __fort_prof_mflag = __fort_prof_init();
  __fort_trac_mflag = __fort_trac_init();
  __fort_entry_mflag = __fort_stat_mflag | __fort_prof_mflag | __fort_trac_mflag;
}

/* FIXME: still used ?*/
/* function entry  */

void ENTFTN(FUNCTION_ENTRYA, function_entrya)(__INT_T *line, __INT_T *lines,
                                            DCHAR(func),
                                            DCHAR(file) DCLEN64(func) DCLEN64(file))
{
  int cline; /* calling line number */
  int n;

  cline = *LINENO;
  pentc->cline = *LINENO; /* save current line number on stack */
  pentc++;                /* next frame */
  if (pentc == pente) {
    n = pente - pentb;
    pentb =
        (struct pent *)__fort_realloc(pentb, (n + PENTS) * sizeof(struct pent));
    pentc = pentb + n;
    pente = pentb + n + PENTS;
  }
  pentc->func = CADR(func);
  pentc->funcl = CLEN(func);
  pentc->file = CADR(file);
  pentc->filel = CLEN(file);
  *LINENO = *line;
  pentc->line = *line;
  pentc->lines = *lines;
  __fort_stat_function_entry(*line, *lines, cline, CADR(func), CADR(file),
                            CLEN(func), CLEN(file));
  __fort_prof_function_entry(*line, *lines, cline, CADR(func), CADR(file),
                            CLEN(func), CLEN(file));
  __fort_trac_function_entry(*line, *lines, cline, CADR(func), CADR(file),
                            CLEN(func), CLEN(file));
}

/* 32 bit CLEN version */
void ENTFTN(FUNCTION_ENTRY, function_entry)(__INT_T *line, __INT_T *lines,
                                            DCHAR(func),
                                            DCHAR(file) DCLEN(func) DCLEN(file))
{
  ENTFTN(FUNCTION_ENTRYA, function_entrya)(line, lines, CADR(func), CADR(file),
                                           (__CLEN_T)CLEN(func), (__CLEN_T)CLEN(file));
}

/*FIXME: still used */
/* function exit  */

void ENTFTN(FUNCTION_EXIT, function_exit)()
{
  __fort_stat_function_exit();
  __fort_prof_function_exit();
  __fort_trac_function_exit();
  pentc--;
  *LINENO = pentc->cline; /* restore current line number */
}

/*FIXME: still used */
/* line entry  */

void ENTFTN(LINE_ENTRY, line_entry)()
{
  __fort_stat_line_entry(*LINENO);
  __fort_prof_line_entry(*LINENO);
  __fort_trac_line_entry(*LINENO);
}

/* print traceback */

void
__fort_traceback()
{
  struct pent *pe;
  char buf[512];
  char *p;
  int lcpu;

  if (pentb == (struct pent *)0) {
    return;
  }
  if (pentc == pentb) {
    return;
  }
  pentc->cline = *LINENO; /* set current line */
  pe = pentc;
  lcpu = GET_DIST_LCPU;
  sprintf(buf, "%d: Traceback:\n", lcpu);
  write(2, buf, strlen(buf));
  while (pe > pentb) {
    p = buf;
    sprintf(p, "%d:   ", lcpu);
    p += strlen(p);
    strncpy(p, pe->func, pe->funcl);
    p += pe->funcl;
    sprintf(p, " at line %d in file \"", pe->cline);
    p += strlen(p);
    strncpy(p, pe->file, pe->filel);
    p += pe->filel;
    strcpy(p, "\"\n");
    write(2, buf, strlen(buf));
    pe--;
  }
}

void ENTFTN(TRACEBACK, traceback)() { __fort_traceback(); }

/* print message with one level call traceback */

void
__fort_tracecall(const char *msg)
{
  struct pent *pe;
  char buf[512];
  char *p;

  p = buf;
  sprintf(p, "%d: %s", GET_DIST_LCPU, msg);
  p += strlen(p);

  pe = pentc;
  if (pentb && pe > pentb) {
    strcpy(p, " in ");
    p += 4;
    strncpy(p, pe->func, pe->funcl);
    p += pe->funcl;
    strcpy(p, " at \"");
    p += 5;
    strncpy(p, pe->file, pe->filel);
    p += pe->filel;
    sprintf(p, "\":%d", *LINENO);
    p += strlen(p);
    if (--pe > pentb) {
      strcpy(p, " called from ");
      p += 13;
      strncpy(p, pe->func, pe->funcl);
      p += pe->funcl;
      strcpy(p, " at \"");
      p += 5;
      strncpy(p, pe->file, pe->filel);
      p += pe->filel;
      sprintf(p, "\":%d", pe->cline);
      p += strlen(p);
    }
  }
  *p++ = '\n';
  *p++ = '\0';
  write(2, buf, strlen(buf));
}

void ENTFTN(TRACECALLA, tracecalla)(DCHAR(msg) DCLEN64(msg))
{
  char buf[257];
  __CLEN_T i;
  __CLEN_T len = CLEN(msg);
  char *p = CADR(msg);
  if (len > 256)
    len = 256;
  for (i = 0; i < len; ++i)
    buf[i] = p[i];
  buf[len] = '\0';
  __fort_tracecall(buf);
}

/* 32 bit CLEN version */
void ENTFTN(TRACECALL, tracecall)(DCHAR(msg) DCLEN(msg))
{
  ENTFTN(TRACECALLA, tracecalla)(CADR(msg), (__CLEN_T)CLEN(msg));
}

/* update start receive message stats */

void
__fort_entry_recv(int cpu, long len)
{
  if (__fort_stat_mflag) {
    __fort_stat_recv(cpu, len);
  }
  if (__fort_prof_mflag) {
    __fort_prof_recv(cpu, len);
  }
  if (__fort_trac_mflag) {
    __fort_trac_recv(cpu, len);
  }
}

/* update done receive message stats */

void
__fort_entry_recv_done(int cpu)
{
  if (__fort_stat_mflag) {
    __fort_stat_recv_done(cpu);
  }
  if (__fort_prof_mflag) {
    __fort_prof_recv_done(cpu);
  }
  if (__fort_trac_mflag) {
    __fort_trac_recv_done(cpu);
  }
}

/* update start send message stats */

void
__fort_entry_send(int cpu, long len)
{
  if (__fort_stat_mflag) {
    __fort_stat_send(cpu, len);
  }
  if (__fort_prof_mflag) {
    __fort_prof_send(cpu, len);
  }
  if (__fort_trac_mflag) {
    __fort_trac_send(cpu, len);
  }
}

/* update done send message stats */

void
__fort_entry_send_done(int cpu)
{
  if (__fort_stat_mflag) {
    __fort_stat_send_done(cpu);
  }
  if (__fort_prof_mflag) {
    __fort_prof_send_done(cpu);
  }
  if (__fort_trac_mflag) {
    __fort_trac_send_done(cpu);
  }
}

/* update start copy message stats */

void
__fort_entry_copy(long len)
{
  if (__fort_stat_mflag) {
    __fort_stat_copy(len);
  }
  if (__fort_prof_mflag) {
    __fort_prof_copy(len);
  }
  if (__fort_trac_mflag) {
    __fort_trac_copy(len);
  }
}

/* update done copy message stats */

void
__fort_entry_copy_done(void)
{
  if (__fort_stat_mflag) {
    __fort_stat_copy_done();
  }
  if (__fort_prof_mflag) {
    __fort_prof_copy_done();
  }
  if (__fort_trac_mflag) {
    __fort_trac_copy_done();
  }
}

/* update start asynch receive message stats */

void
__fort_entry_arecv(int cpu, long len, int reqn)
{
  if (__fort_stat_mflag) {
    __fort_stat_arecv(cpu, len, reqn);
  }
  if (__fort_prof_mflag) {
    __fort_prof_arecv(cpu, len, reqn);
  }
  if (__fort_trac_mflag) {
    __fort_trac_arecv(cpu, len, reqn);
  }
}

/* update done asynch receive message stats */

void
__fort_entry_arecv_done(int cpu)
{
  if (__fort_stat_mflag) {
    __fort_stat_arecv_done(cpu);
  }
  if (__fort_prof_mflag) {
    __fort_prof_arecv_done(cpu);
  }
  if (__fort_trac_mflag) {
    __fort_trac_recv_done(cpu);
  }
}

/* update start asynch send message stats */

void
__fort_entry_asend(int cpu, long len, int reqn)
{
  if (__fort_stat_mflag) {
    __fort_stat_asend(cpu, len, reqn);
  }
  if (__fort_prof_mflag) {
    __fort_prof_asend(cpu, len, reqn);
  }
  if (__fort_trac_mflag) {
    __fort_trac_asend(cpu, len, reqn);
  }
}

/* update done asynch send message stats */

void
__fort_entry_asend_done(int cpu)
{
  if (__fort_stat_mflag) {
    __fort_stat_asend_done(cpu);
  }
  if (__fort_prof_mflag) {
    __fort_prof_asend_done(cpu);
  }
  if (__fort_trac_mflag) {
    __fort_trac_asend_done(cpu);
  }
}

/* update start asynch wait message stats */

void
__fort_entry_await(int reqn)
{
  if (__fort_stat_mflag) {
    __fort_stat_await(reqn);
  }
  if (__fort_prof_mflag) {
    __fort_prof_await(reqn);
  }
  if (__fort_trac_mflag) {
    __fort_trac_await(reqn);
  }
}

/* update done async wait message stats */

void
__fort_entry_await_done(int reqn)
{
  if (__fort_stat_mflag) {
    __fort_stat_await_done(reqn);
  }
  if (__fort_prof_mflag) {
    __fort_prof_await_done(reqn);
  }
  if (__fort_trac_mflag) {
    __fort_trac_await_done(reqn);
  }
}

/** \brief Termination */
void
__fort_entry_term(void)
{
  __fort_entry_mflag = 0; /* don't count overhead */
  __fort_stat_term();
  __fort_prof_term();
  __fort_trac_term();
}
