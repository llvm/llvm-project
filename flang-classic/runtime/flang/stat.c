/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/* clang-format off */

#include "stdioInterf.h"
#include "fioMacros.h"
#include "timeBlk.h"
#include <string.h>
#include <memory.h>

#if defined(_WIN64)
#include <io.h> // for _write
#define write _write
#endif

/* defined in stat_linux.c */
void __fort_gettb(struct tb *);

static struct tb tb0 = {/* stats at beginning of program */
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ""};

static struct tb tb1 = {/* stats at end of program */
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ""};

/* Fortran-callable time routine
 * calling:
 *           real*8 times(3)
 *           call ENTFTN(TIMES,times)(times)
 * returns:
 *    times(1) = wallclock time
 *    times(2) = user process run time
 *    times(3) = system time
 */

void ENTFTN(TIMES, times)(double *times)
{
  struct tb t;

  __fort_gettb(&t);
  times[0] = t.r;
  times[1] = t.u;
  times[2] = t.s;
}

/** \brief Begin stats */
int
__fort_stat_init(void)
{
  __fort_gettb(&tb0);
  return (1);
}

/* scale byte quantity */

static const char *
scale_bytes(double d, double *ds)
{
  const char *s;

  s = "B";
  if (d >= 1024) {
    d = (d + 1023) / 1024;
    s = "KB";
  }
  if (d >= 1024) {
    d = (d + 1023) / 1024;
    s = "MB";
  }
  if (d >= 1024) {
    d = (d + 1023) / 1024;
    s = "GB";
  }
  if (d >= 1024) {
    d = (d + 1023) / 1024;
    s = "TB";
  }
  *ds = d;
  return s;
}

/* scale byte quantity, beginning with kilobytes */

static const char *
scale_kbytes(double d, double *ds)
{
  const char *s;

  d = (d + 1023) / 1024;
  s = "KB";
  if (d >= 1024) {
    d = (d + 1023) / 1024;
    s = "MB";
  }
  if (d >= 1024) {
    d = (d + 1023) / 1024;
    s = "GB";
  }
  if (d >= 1024) {
    d = (d + 1023) / 1024;
    s = "TB";
  }
  *ds = d;
  return s;
}

/* print cpu info */

static void cpu(struct tb *tbp)
{
  int i, quiet, tcpus;
  struct tb n, a, x, e;
  char buf[256];
  double d;

  tcpus = GET_DIST_TCPUS;
  fprintf(__io_stderr(), "\n");
  fprintf(__io_stderr(),
          "cpu        real      user       sys     ratio   node\n");
  n.r = x.r = e.r = tbp[0].r;
  n.u = x.u = e.u = tbp[0].u;
  n.s = x.s = e.s = tbp[0].s;
  for (i = 1; i < tcpus; i++) {
    if (tbp[i].r < n.r)
      n.r = tbp[i].r;
    if (tbp[i].r > x.r)
      x.r = tbp[i].r;
    if (tbp[i].u < n.u)
      n.u = tbp[i].u;
    if (tbp[i].u > x.u)
      x.u = tbp[i].u;
    if (tbp[i].s < n.s)
      n.s = tbp[i].s;
    if (tbp[i].s > x.s)
      x.s = tbp[i].s;
    e.r += tbp[i].r;
    e.u += tbp[i].u;
    e.s += tbp[i].s;
  }

  quiet = GET_DIST_QUIET;
  if (quiet & Q_CPUS) {
    for (i = 0; i < tcpus; i++) {
      d = (tbp[i].r == 0.0 ? 0.0 : (tbp[i].u + tbp[i].s) / tbp[i].r);
      sprintf(buf, "%4d%c%10.2f%10.2f%10.2f%9.0f%%   %-s\n", i,
              (i == GET_DIST_IOPROC ? '*' : ' '), tbp[i].r, tbp[i].u, tbp[i].s,
              d * 100, tbp[i].host);
      write(2, buf, strlen(buf));
    }
  }
  if ((quiet & Q_CPUS) && (tcpus > 1)) {
    a.r = e.r / (float)tcpus;
    a.u = e.u / (float)tcpus;
    a.s = e.s / (float)tcpus;
    sprintf(buf, " min %10.2f%10.2f%10.2f\n", n.r, n.u, n.s);
    write(2, buf, strlen(buf));
    sprintf(buf, " avg %10.2f%10.2f%10.2f\n", a.r, a.u, a.s);
    write(2, buf, strlen(buf));
    sprintf(buf, " max %10.2f%10.2f%10.2f\n", x.r, x.u, x.s);
    write(2, buf, strlen(buf));
  }
  d = (x.r == 0.0 ? 0.0 : (e.u + e.s) / x.r);
  sprintf(buf, "total%10.2f%10.2f%10.2f%9.2fx\n", x.r, e.u, e.s, d);
  write(2, buf, strlen(buf));
}

/* print memory info */

static void mem(struct tb *tbp)
{
  double tmaxrss; /* total max set size */
  double tminflt; /* total minor fault */
  double tmajflt; /* total major fault */
  double tnvcsw;  /* total voluntary switches */
  double tnivcsw; /* total involuntary switches */
  double tsbrk;   /* total heap used (local) */
  double tgsbrk;  /* total heap used (global) */
  int i, quiet, tcpus;
  const char *s_sbrk, *s_gsbrk;
  double d_sbrk, d_gsbrk;
  char buf[256];

  fprintf(__io_stderr(), "\n");
  fprintf(__io_stderr(), "memory    local    global  res size  pag flts  pag "
                           "flts voluntary  involunt\n");
  fprintf(__io_stderr(), "           heap      heap   (pages)     minor     "
                           "major  switches  switches\n");
  tmaxrss = 0.0;
  tminflt = 0.0;
  tmajflt = 0.0;
  tnvcsw = 0.0;
  tnivcsw = 0.0;
  tsbrk = 0.0;
  tgsbrk = 0.0;
  quiet = GET_DIST_QUIET;
  tcpus = GET_DIST_TCPUS;
  for (i = 0; i < tcpus; i++) {
    tmaxrss += tbp[i].maxrss;
    tminflt += tbp[i].minflt;
    tmajflt += tbp[i].majflt;
    tnvcsw += tbp[i].nvcsw;
    tnivcsw += tbp[i].nivcsw;
    tsbrk += tbp[i].sbrk;
    tgsbrk += tbp[i].gsbrk;
    if (quiet & Q_MEMS) {
      s_sbrk = scale_kbytes(tbp[i].sbrk, &d_sbrk);
      s_gsbrk = scale_kbytes(tbp[i].gsbrk, &d_gsbrk);
      sprintf(buf,
              "%4d%c%8.0lf%2s%8.0lf%2s%10.0lf%10.0lf%10.0lf%10.0lf%10.0lf\n", i,
              (i == GET_DIST_IOPROC ? '*' : ' '), d_sbrk, s_sbrk, d_gsbrk,
              s_gsbrk, tbp[i].maxrss, tbp[i].minflt, tbp[i].majflt,
              tbp[i].nvcsw, tbp[i].nivcsw);
      write(2, buf, strlen(buf));
    }
  }
  s_sbrk = scale_kbytes(tsbrk, &d_sbrk);
  s_gsbrk = scale_kbytes(tgsbrk, &d_gsbrk);
  sprintf(buf, "total%8.0lf%2s%8.0lf%2s%10.0lf%10.0lf%10.0lf%10.0lf%10.0lf\n",
          d_sbrk, s_sbrk, d_gsbrk, s_gsbrk, tmaxrss, tminflt, tmajflt, tnvcsw,
          tnivcsw);
  write(2, buf, strlen(buf));
}

/* print message info */

static void msg(struct tb *tbp)
{
  int i, quiet, tcpus;
  double ds, dr, dc, dst, drt, dct;
  double mst, mrt, mct, ast, art, act;
  const char *ss, *sr, *sc, *as, *ar, *ac;
  double d;
  char buf[256];

  fprintf(__io_stderr(), "\n");
  fprintf(__io_stderr(), "messages  send   send   send     recv   recv   "
                           "recv     copy   copy   copy\n");
  fprintf(__io_stderr(), "           cnt  total    avg      cnt  total    "
                           "avg      cnt  total    avg\n");
  dst = 0;
  drt = 0;
  dct = 0;
  mst = 0;
  mrt = 0;
  mct = 0;
  quiet = GET_DIST_QUIET;
  tcpus = GET_DIST_TCPUS;
  for (i = 0; i < tcpus; i++) {
    dst += tbp[i].bytes;
    drt += tbp[i].byter;
    dct += tbp[i].bytec;
    mst += tbp[i].datas;
    mrt += tbp[i].datar;
    mct += tbp[i].datac;
    if (quiet & Q_MSGS) {
      ss = scale_bytes(tbp[i].bytes, &ds);
      sr = scale_bytes(tbp[i].byter, &dr);
      sc = scale_bytes(tbp[i].bytec, &dc);
      d = (tbp[i].datas == 0 ? 0.0 : tbp[i].bytes / tbp[i].datas);
      as = scale_bytes(d, &ast);
      d = (tbp[i].datar == 0 ? 0.0 : tbp[i].byter / tbp[i].datar);
      ar = scale_bytes(d, &art);
      d = (tbp[i].datac == 0 ? 0.0 : tbp[i].bytec / tbp[i].datac);
      ac = scale_bytes(d, &act);
      sprintf(buf, "%4d%c%9.0lf%5.0lf%2s%5.0lf%2s%9.0lf%5.0lf%2s%5.0lf%2s%9."
                   "0lf%5.0lf%2s%5.0lf%2s\n",
              i, (i == GET_DIST_IOPROC ? '*' : ' '), tbp[i].datas, ds, ss, ast,
              as, tbp[i].datar, dr, sr, art, ar, tbp[i].datac, dc, sc, act, ac);
      write(2, buf, strlen(buf));
    }
  }
  ss = scale_bytes(dst, &ds);
  sr = scale_bytes(drt, &dr);
  sc = scale_bytes(dct, &dc);
  d = (dst == 0 ? 0.0 : dst / mst);
  as = scale_bytes(d, &ast);
  d = (drt == 0 ? 0.0 : drt / mrt);
  ar = scale_bytes(d, &art);
  d = (dct == 0 ? 0.0 : dct / mct);
  ac = scale_bytes(d, &act);
  sprintf(buf, "total%9.0lf%5.0lf%2s%5.0lf%2s%9.0lf%5.0lf%2s%5.0lf%2s%9.0lf%5."
               "0lf%2s%5.0lf%2s\n",
          mst, ds, ss, ast, as, mrt, dr, sr, art, ar, mct, dc, sc, act, ac);
  write(2, buf, strlen(buf));
}

/** \brief Print stats (called by all cpus) */
void
__fort_stat_term(void)
{
  int i, ioproc, quiet, tcpus;
  struct tb *tbp;

  __fort_gettb(&tb1);
  tb1.r -= tb0.r;
  tb1.u -= tb0.u;
  tb1.s -= tb0.s;
  if (tb1.r < (tb1.u + tb1.s)) {
    tb1.r = tb1.u + tb1.s;
  }
  tb1.sbrk -= tb0.sbrk;
  tb1.gsbrk -= tb0.gsbrk;

  tcpus = GET_DIST_TCPUS;
  tbp = (struct tb *)__fort_gmalloc(sizeof(struct tb) * tcpus);

  ioproc = GET_DIST_IOPROC;
  if (__fort_is_ioproc() == 0) { /* not i/o process */
    __fort_rsend(ioproc, (char *)&(tb1), sizeof(struct tb), 1, __UCHAR);
  } else {
    for (i = 0; i < tcpus; i++) { /* i/o process */
      if (i == ioproc) {
        continue;
      }
      __fort_rrecv(i, (char *)&(tbp[i]), sizeof(struct tb), 1, __UCHAR);
    }
    tbp[ioproc] = tb1;
    quiet = GET_DIST_QUIET;
    if (quiet & (Q_CPU | Q_CPUS)) {
      cpu(tbp);
    }
    if (quiet & (Q_MEM | Q_MEMS)) {
      mem(tbp);
    }
    if (quiet & (Q_MSG | Q_MSGS)) {
      msg(tbp);
    }
  }

  __fort_gfree(tbp);
}

/* update start receive message stats */

void
__fort_stat_recv(int cpu, long len)
/* cpu: sending cpu */
/* len: total length in bytes */
{
  tb1.datar++;
  tb1.byter += len;
}

/** \brief Update done receive message stats */
void
__fort_stat_recv_done(int cpu /* sending cpu */)
{
}

/** \brief Update start send message stats
 * \param cpu: receiving cpu
 * \param len: total length in bytes
 */
void
__fort_stat_send(int cpu, long len)
{
  tb1.datas++;
  tb1.bytes += len;
}

/** \brief Update done send message stats
 * \param cpu - receiving cpu
 */
void
__fort_stat_send_done(int cpu)
{
}

/** \brief Update start bcopy message stats
 * \param len: total length in bytes
 */
void
__fort_stat_copy(long len)
{
  tb1.datac++;
  tb1.bytec += len;
}

/** \brief Update done bcopy message stats */
void
__fort_stat_copy_done(void)
{
}

/** \brief Update start asynch receive message stats */
void
__fort_stat_arecv(int cpu, long len, int reqn)
{
  tb1.datar += 1;
  tb1.byter += len;
}

/* update done asynch receive message stats */

void
__fort_stat_arecv_done(int cpu)
{
}

/* update start asynch send message stats */

void
__fort_stat_asend(int cpu, long len, int reqn)
{
  tb1.datas += 1;
  tb1.bytes += len;
}

/* update done asynch send message stats */

void
__fort_stat_asend_done(int cpu)
{
}

/* update start await receive message stats */

void
__fort_stat_await(int reqn)
{
}

/* update done await receive message stats */

void
__fort_stat_await_done(int reqn)
{
}

/* return incremental stats (internal use only) */

void ENTFTN(MSGSTATS, msgstats)(__INT_T *msgstats)
{
  msgstats[0] = tb0.datas; /* send count   */
  msgstats[1] = tb0.datar; /* recv count   */
  msgstats[2] = tb0.bytes; /* bytes sent   */
  msgstats[3] = tb0.byter; /* bytes recv'd */
}

/* FIXME: still used? */
/** \brief Function entry  */
void
__fort_stat_function_entry(int line, int lines, int cline, char *func,
                          char *file, int funcl, int filel)
{
}

/* FIXME: still used? */
/** \brief Line entry  */
void
__fort_stat_line_entry(int line /* current line number */)
{
}

/* FIXME: still used? */
/** \brief Function exit  */
void
__fort_stat_function_exit(void)
{
}
