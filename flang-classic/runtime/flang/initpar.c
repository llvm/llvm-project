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

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <memory.h>
#if !defined(_WIN64)
#include <sys/time.h>
#endif

#include "global.h"
#include "stdioInterf.h"
/* FIXME: HACK
 * include/stdioInterf.h: #define __fort_getenv(name) getenv(name)
 * changes the name of __fort_getenv defined in this file.
 */
#undef __fort_getenv

#include "fioMacros.h"

#include "open_close.h"
#include "format.h"


#if   defined(TARGET_OSX)
#include <crt_externs.h>
#elif defined(_WIN64)
/* OPENTOOLS14 has changed the name.  wrap _environ for all of windowws */
char **__io_environ();
#else
WIN_MSVCRT_IMP char **environ;
#endif
#include "fort_vars.h"
char *__fort_getgbuf(long);
extern void __fort_init_consts();

void __fort_print_version();                         /* version.c */

extern int __io_get_argc();
extern char **__io_get_argv();
extern void __io_set_argc(int);

static char **arg; /* first arg */
static char **env; /* internal version of environ */

#define MAXOPTS 128
static char *opts[MAXOPTS]; 
static char *optarea;       /* malloc'ed area for opts */

static struct {
  char consts;
  char atexit;
} inited;

/* common blocks containing values for inlined number_of_processors()
   and my_processor() functions */

WIN_API __INT_T ENTCOMN(NP, np)[];
WIN_API __INT_T ENTCOMN(ME, me)[];

#if defined(_WIN64)
#define write _write
#endif

/* Return logical cpu number */

int
__fort_myprocnum()
{
  return (__fort_lcpu); /* non-shared-memory version */
}

/* Return total number of processors */

int
__fort_ncpus()
{
  return __fort_tcpus;
}

#if defined(_WIN64)
__INT_T *CORMEM;

/* special argument pointer access routines */

char *
__get_fort_01_addr(void)
{
  return (char *)ENTCOMN(0, 0);
}

char *
__get_fort_02_addr(void)
{
  return (char *)ENTCOMN(0, 0) + 4;
}

char *
__get_fort_03_addr(void)
{
  return (char *)ENTCOMN(0, 0) + 8;
}

char *
__get_fort_04_addr(void)
{
  return (char *)ENTCOMN(0, 0) + 12;
}

char *
__get_fort_0c_addr(void)
{
  return (char *)ENTCOMN(0C, 0c);
}

void
__set_fort_0l_addr(int *addr)
{
}

void
__CORMEM_SCAN(void)
{
}

char *
__get_fort_local_mode_addr(void)
{
  return (char *)ENTCOMN(LOCAL_MODE, local_mode);
}

char *
__get_fort_me_addr(void)
{
  return (char *)ENTCOMN(ME, me);
}

char *
__get_fort_np_addr(void)
{
  return (char *)ENTCOMN(NP, np);
}

/* access routines for data shared between windows dlls */

/* logical CPU id of the i/o processor */

int
__get_fort_debug(void)
{
  return __fort_debug;
}

void
__set_fort_debug(int debug)
{
  __fort_debug = debug;
}

int
__get_fort_debugn(void)
{
  return __fort_debugn;
}

void
__set_fort_debugn(int debugn)
{
  __fort_debugn = debugn;
}

long
__get_fort_heapz(void)
{
  return __fort_heapz;
}

void
__set_fort_heapz(long heapz)
{
  __fort_heapz = heapz;
}

int
__get_fort_ioproc(void)
{
  return __fort_ioproc;
}

void
__set_fort_ioproc(int ioproc)
{
  __fort_ioproc = ioproc;
}

/* logical cpu number */

int
__get_fort_lcpu(void)
{
  return __fort_lcpu;
}

void
__set_fort_lcpu(int lcpu)
{
  __fort_lcpu = lcpu;
}

int *
__get_fort_lcpu_addr(void)
{
  return &__fort_lcpu;
}

/* pario */

int
__get_fort_pario(void)
{
  return __fort_pario;
}

void
__set_fort_pario(int pario)
{
  __fort_pario = pario;
}

/* runtime statistics */

int
__get_fort_quiet(void)
{
  return __fort_quiet;
}

void
__set_fort_quiet(int quiet)
{
  __fort_quiet = quiet;
}

/* total number of processors */

int
__get_fort_tcpus(void)
{
  return __fort_tcpus;
}

int *
__get_fort_tcpus_addr(void)
{
  return &__fort_tcpus;
}

void
__set_fort_tcpus(int tcpus)
{
  __fort_tcpus = tcpus;
}

/* tid for each processor   */

int *
__get_fort_tids(void)
{
  return __fort_tids;
}

void
__set_fort_tids(int *tids)
{
  __fort_tids = tids;
}

int
__get_fort_tids_elem(int idx)
{
  return __fort_tids[idx];
}

void
__set_fort_tids_elem(int idx, int val)
{
  __fort_tids[idx] = val;
}

#endif /* _WIN64 */

int
__fort_getioproc()
{
  return (__fort_ioproc);
}

/* Return true if this is the i/o processor. */

int
__fort_is_ioproc()
{
  return (__fort_lcpu == __fort_ioproc);
}

/* abort with message */

void
__fort_abort(const char *s)
{
  char buf[256];

  if (s != NULL) {
    sprintf(buf, "%d: %s\n", __fort_lcpu, s);
    write(2, buf, strlen(buf));
  }
  __fort_abortx();
}

/* abort with perror message */

void
__fort_abortp(const char *s)
{
  fprintf(__io_stderr(), "%d: ", __fort_lcpu);
  perror(s);
  __fort_abort(NULL);
}

/* exit */

void
__fort_exit(int s)
{
  exit(s);
}

/* init command line processing */

static char *dumarg = NULL;

static void
__fort_initarg()
{
  char **v;

  if (arg != (char **)0) {
    return;
  }
  v = __io_get_argv();
  if (v == (char **)0) {
    arg = &(dumarg); /* no argv -> no args */
  } else {
    arg = v;
  }
#if   defined(TARGET_OSX)
  env = *_NSGetEnviron();
#elif defined(_WIN64)
  env = __io_environ();
#else
  env = environ;
#endif
}

/** \brief getenv (uses env, not environ) */
char *
__fort_getenv(const char *nm)
{
  char **e;
  int n;

  n = strlen(nm);
#if defined(TARGET_OSX)
  e = env;
#else             
  e = environ;
#endif
  while (*e != NULL) {
    if ((strncmp(*e, nm, n) == 0) && ((*((*e) + n)) == '=')) {
      return ((*e) + n + 1);
    }
    e++;
  }
  return (NULL);
}

/* init option processing */

static void
__fort_initopt()
{
  char *p, *q;
  int i;

  p = __fort_getenv("PGDIST_OPTS");
  if (p == NULL) {
    return;
  }
  if (optarea != NULL) {
    __fort_free(optarea);
  }
  optarea = __fort_malloc(strlen(p) + 1);
  q = optarea;
  strcpy(q, p);
  i = 0;
  while (1) {
    while (*q == ' ') {
      q++;
    }
    if (*q == '\0') {
      break;
    }
    if (i >= (MAXOPTS - 1)) {
      __fort_abort("PGDIST_OPTS: too many options");
    }
    opts[i++] = q;
    while ((*q != ' ') && (*q != '\0')) {
      q++;
    }
    if (*q == ' ') {
      *q++ = '\0';
    }
  }
  opts[i] = NULL;
}

/* get option (command line -xx and environment */
const char
*__fort_getopt(const char *opt)
{
  char env[64];
  char *p;
  const char *q;
  int n;

  if (arg == NULL)
    return NULL;
  p = NULL;
  for (n = 0; arg[n] != NULL; n++) {
    if (strcmp(arg[n], opt) == 0) {
      p = arg[n + 1];
      if (p == NULL) {
        return "";
      }
      break;
    }
  }
  if (p == NULL) {
    strcpy(env, "PGHPF_");
    p = env + 6;
    q = opt + 1;
    while (*q != '\0') {
      *p++ = toupper(*q++);
    }
    *p++ = '\0';
    p = __fort_getenv(env);
  }
  if (p == NULL) {
    for (n = 0; opts[n] != NULL; n++) {
      if (strcmp(opts[n], opt) == 0) {
        p = opts[n + 1];
        if (p == NULL) {
          return "";
        }
        break;
      }
    }
  }
  if ((strcmp(opt, "-g") == 0) && (p != NULL) && (*p == '-')) {
    return "";
  }
  return p;
}

/* abort because of problem with command/environment option */

static void
getopt_abort(const char *problem, const char *opt)
{
  char buf[128], *p;
  const char *q;

  p = buf;
  q = opt;

  while (*++q != '\0')
    *p++ = toupper(*q);
  *p++ = '\0';
  sprintf(p, "%s for %s/%s command/environment option\n", problem, opt,
          buf);
  __fort_abort(p);
}

/* get numeric option */

long
__fort_getoptn(const char *opt, long def)
{
  const char *p;
  char *q;
  long n;

  p = __fort_getopt(opt);
  if (p == NULL)
    return def; /* default if option is absent */
  n = __fort_strtol(p, &q, 0);
  if (q == p || *q != '\0')
    getopt_abort("missing or invalid numeric value", opt);
  return n;
}

/* get yes/no option */

int
__fort_getoptb(const char *opt, int def)
{
  const char *p;
  int n = 0;

  p = __fort_getopt(opt);
  if (p == NULL)
    return def; /* default if option is absent */
  if (*p == 'y' || *p == 'Y')
    n = 1;
  else if (*p == 'n' || *p == 'N')
    n = 0;
  else
    getopt_abort("missing or invalid yes/no value", opt);
  return n;
}

/* init stats (set options) */

static void
__fort_istat()
{
  const char *p;

  p = __fort_getopt("-stat");
  if (p == NULL) {
    return;
  }
  if ((*p == '\0') || (*p == '-')) {
    p = "all";
  }
  while (1) {
    if (strncmp(p, "cpus", 4) == 0) {
      __fort_quiet |= Q_CPUS;
    } else if (strncmp(p, "mems", 4) == 0) {
      __fort_quiet |= Q_MEMS;
    } else if (strncmp(p, "msgs", 4) == 0) {
      __fort_quiet |= Q_MSGS;
    } else if (strncmp(p, "alls", 4) == 0) {
      __fort_quiet |= (Q_CPUS | Q_MEMS | Q_MSGS);
    } else if (strncmp(p, "cpu", 3) == 0) {
      __fort_quiet |= Q_CPU;
    } else if (strncmp(p, "mem", 3) == 0) {
      __fort_quiet |= Q_MEM;
    } else if (strncmp(p, "msg", 3) == 0) {
      __fort_quiet |= Q_MSG;
    } else if (strncmp(p, "all", 3) == 0) {
      __fort_quiet |= (Q_CPU | Q_MEM | Q_MSG);
    } else if (strncmp(p, "prof", 4) == 0) {
      __fort_quiet |= Q_PROF;
    } else if (strncmp(p, "trace", 5) == 0) {
      __fort_quiet |= Q_TRAC;
    } else if ((*p >= '0') && (*p <= '9')) {
      __fort_quiet |= (int)strtol(p, (char **)0, 0);
    } else {
      getopt_abort("invalid format", "-stat");
    }
    p = strchr(p, ',');
    if (p == NULL) {
      break;
    }
    p++;
  }
}

/* process (what used to be) generic command/environment options */

static void
__fort_initcom()
{
  const char *p;
  char *q;
  int n;

  /* -test [<n>] */

  p = __fort_getopt("-test");
  if (p) {
    __fort_test = (int)__fort_strtol(p, &q, 0);
    if (q == p)
      __fort_test = -1;
    else if (*q != '\0')
      getopt_abort("invalid numeric value", "-test");
  }

  /* -np <n> = number of processors */

  p = __fort_getopt("-np");
  if (p) {
    n = (int)__fort_strtol(p, &q, 0);
    if (q == p || *q != '\0' || n < 1)
      getopt_abort("missing or invalid numeric value", "-np");
    __fort_tcpus = n;
  }

  /* -g [<n>|all] = debug */

  p = __fort_getopt("-g");
  if (p) {
    __fort_debug = 1;
    __fort_debugn = (int)__fort_strtol(p, &q, 0);
    if (q == p)
      __fort_debugn = -1;
    else if (*q != '\0' || __fort_debugn < 0 || __fort_debugn >= __fort_tcpus)
      getopt_abort("invalid numeric value", "-g");
  }

  /* -stat ... */

  __fort_istat();

  /* -prof av[erage]|no[ne]|al[l] */

  p = __fort_getopt("-prof");
  if (p) {
    int k = strlen(p);
    if (k < 2)
      k = 2;
    if (strncmp(p, "average", k) == 0)
      __fort_quiet |= Q_PROF_AVG;
    else if (strncmp(p, "none", k) == 0)
      __fort_quiet |= Q_PROF_NONE;
    else if (strncmp(p, "all", k) != 0)
      getopt_abort("invalid value", "-prof");
  }
}

/* init and process command/environment options */

void
__fort_procargs()
{

  if (arg != (char **)0) {
    return;
  }
  __fort_initarg(); /* init command line args */
  __fort_initopt(); /* init opt */
  __fort_initcom(); /* init common arg/env */
}

/* pass an arg to other processors, passing of a null is permitted */

static char *
__fort_passarg(int fr, int tol, int toh, char *val)
{
  int cpu;
  int len;
  char *p;

  if (__fort_lcpu == fr) {
    len = (val != NULL ? strlen(val) + 1 : 0);
    for (cpu = tol; cpu < toh; cpu++) {
      __fort_rsendl(cpu, &len, sizeof(len), 1, __UCHAR, 1);
      if (len != 0) {
        __fort_rsendl(cpu, val, len, 1, __UCHAR, 1);
      }
    }
    p = val;
  } else {
    __fort_rrecvl(fr, &len, sizeof(len), 1, __UCHAR, 1);
    if (len == 0) {
      p = NULL;
    } else {
      p = __fort_malloc(len);
      __fort_rrecvl(fr, p, len, 1, __UCHAR, 1);
    }
  }
  return (p);
}

/* pass arglist */

void
__fort_passargs(int fr, int tol, int toh)
{
  char **toe;
  char **fre;
  int n;
  int cpu;

  if (__fort_lcpu == fr) {
    n = 0;
    while (env[n] != NULL) {
      n++;
    }
    n++;
    for (cpu = tol; cpu < toh; cpu++) {
      __fort_rsendl(cpu, &n, sizeof(n), 1, __UCHAR, 1);
    }
    fre = env;
    while (*fre != NULL) {
      if ((strlen(*fre) > 6) && (strncmp("PGHPF_", *fre, 6) == 0)) {
        __fort_passarg(fr, tol, toh, *fre);
      }
      fre++;
    }
    __fort_passarg(fr, tol, toh, NULL);
  } else {
    __fort_rrecvl(fr, &n, sizeof(n), 1, __UCHAR, 1);
    env = (char **)__fort_malloc(n * sizeof(char *));
    toe = env;
    while (1) {
      *toe = __fort_passarg(fr, tol, toh, NULL);
      if (*toe == NULL) {
        break;
      }
      toe++;
    }
  }

  if (__fort_lcpu == fr) {
    n = 0;
    while (arg[n] != NULL) {
      n++;
    }
    n++;
    for (cpu = tol; cpu < toh; cpu++) {
      __fort_rsendl(cpu, &n, sizeof(n), 1, __UCHAR, 1);
    }
    fre = arg;
    while (*fre != NULL) {
      __fort_passarg(fr, tol, toh, *fre);
      fre++;
    }
    __fort_passarg(fr, tol, toh, NULL);
  } else {
    __fort_rrecvl(fr, &n, sizeof(n), 1, __UCHAR, 1);
    arg = (char **)__fort_malloc(n * sizeof(char *));
    toe = arg;
    while (1) {
      *toe = __fort_passarg(fr, tol, toh, NULL);
      if (*toe == NULL) {
        break;
      }
      toe++;
    }
    __fort_initopt(); /* init opt */
    __fort_initcom(); /* init common arg/env */
  }
}

/* terminate everything */

static void
term()
{
  extern void __f90_allo_term(void);
  __f90_allo_term();
  __fortio_cleanup();  /* cleanup i/o */
  __fort_entry_term(); /* end of profiling/tracing/stats */
  __fort_endpar();     /* TI-specific termination */
}

/* initialize everything */

void ENTFTN(INIT, init)(__INT_T *n)
{
  __fort_setarg();      /* set __argv_save and __argc_save (maybe) */
  if (!inited.consts) {
    __fort_init_consts(); /* constants need initialization */
    inited.consts = 1;
  }
  __fort_begpar(*n);    /* TI-specific initialization */

  /* smallest power of 2 >= number of processors */

  for (__fort_np2 = 1; __fort_np2 < __fort_tcpus; __fort_np2 <<= 1)
    ;

  /* -V or -version */

  if (__fort_lcpu == 0 && (__fort_getopt("-V") || __fort_getopt("-version")))
    __fort_print_version();

  __fort_zmem = __fort_getoptb("-zmem", 0);

  __fort_entry_init(); /* start profiling/tracing/stats */

  if (!inited.atexit) {
    atexit(term); /* register term */
    inited.atexit = 1;
  }

  ENTCOMN(NP, np)[0] = __fort_tcpus; /* for number_of_processors() */
  ENTCOMN(ME, me)[0] = __fort_lcpu;  /* for my_processor() */
}

/* pull in the following code (not really called) */

void
__fort_pull_them_in()
{
  __fort_getgbuf(0);
  __fort_rrecv(0, (char *)0, 0, 0, 0);
  __fort_rsend(0, (char *)0, 0, 0, 0);
  __fort_zopen((char *)0);
}

/* -------------------------------------------------------------------- */

/*
 * this routine is called from .init.  it does limited initialization
 * for f90 routines called from a non-f90 main routine.  argc and
 * argv may not be set.
 */


void
__attribute__((constructor))
f90_compiled()
{
  if (!inited.consts) {
    __fort_tcpus = 1;
    __fort_np2 = 1;
    __fort_init_consts(); /* constants need initialization */
    inited.consts = 1;
  }
  if (!inited.atexit) {
    atexit(term); /* register term */
    inited.atexit = 1;
  }
}
