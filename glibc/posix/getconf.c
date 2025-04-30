/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published
   by the Free Software Foundation; version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#include <unistd.h>
#include <errno.h>
#include <error.h>
#include <libintl.h>
#include <locale.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "../version.h"
#define PACKAGE _libc_intl_domainname

#define NEED_SPEC_ARRAY 1
#include <posix-conf-vars.h>

/* If all of the environments are defined in environments.h, then we don't need
   to bother with doing a runtime check for a specific environment.  */
#if (defined _SC_V6_ILP32_OFF32 \
     && defined _SC_V7_LPBIG_OFFBIG \
     && defined _SC_XBS5_LP64_OFF64 \
     && defined _SC_V6_LP64_OFF64 \
     && defined _SC_V7_ILP32_OFFBIG \
     && defined _SC_V6_LPBIG_OFFBIG \
     && defined _SC_V7_LP64_OFF64 \
     && defined _SC_V7_ILP32_OFF32 \
     && defined _SC_XBS5_LPBIG_OFFBIG \
     && defined _SC_XBS5_ILP32_OFFBIG \
     && defined _SC_V6_ILP32_OFFBIG \
     && defined _SC_XBS5_ILP32_OFF32)
# define ALL_ENVIRONMENTS_DEFINED 1
#endif

struct conf
  {
    const char *name;
    const int call_name;
    const enum { SYSCONF, CONFSTR, PATHCONF } call;
  };

static const struct conf vars[] =
  {
    { "LINK_MAX", _PC_LINK_MAX, PATHCONF },
    { "_POSIX_LINK_MAX", _PC_LINK_MAX, PATHCONF },
    { "MAX_CANON", _PC_MAX_CANON, PATHCONF },
    { "_POSIX_MAX_CANON", _PC_MAX_CANON, PATHCONF },
    { "MAX_INPUT", _PC_MAX_INPUT, PATHCONF },
    { "_POSIX_MAX_INPUT", _PC_MAX_INPUT, PATHCONF },
    { "NAME_MAX", _PC_NAME_MAX, PATHCONF },
    { "_POSIX_NAME_MAX", _PC_NAME_MAX, PATHCONF },
    { "PATH_MAX", _PC_PATH_MAX, PATHCONF },
    { "_POSIX_PATH_MAX", _PC_PATH_MAX, PATHCONF },
    { "PIPE_BUF", _PC_PIPE_BUF, PATHCONF },
    { "_POSIX_PIPE_BUF", _PC_PIPE_BUF, PATHCONF },
    { "SOCK_MAXBUF", _PC_SOCK_MAXBUF, PATHCONF },
    { "_POSIX_ASYNC_IO", _PC_ASYNC_IO, PATHCONF },
    { "_POSIX_CHOWN_RESTRICTED", _PC_CHOWN_RESTRICTED, PATHCONF },
    { "_POSIX_NO_TRUNC", _PC_NO_TRUNC, PATHCONF },
    { "_POSIX_PRIO_IO", _PC_PRIO_IO, PATHCONF },
    { "_POSIX_SYNC_IO", _PC_SYNC_IO, PATHCONF },
    { "_POSIX_VDISABLE", _PC_VDISABLE, PATHCONF },

    { "ARG_MAX", _SC_ARG_MAX, SYSCONF },
    { "ATEXIT_MAX", _SC_ATEXIT_MAX, SYSCONF },
    { "CHAR_BIT", _SC_CHAR_BIT, SYSCONF },
    { "CHAR_MAX", _SC_CHAR_MAX, SYSCONF },
    { "CHAR_MIN", _SC_CHAR_MIN, SYSCONF },
    { "CHILD_MAX", _SC_CHILD_MAX, SYSCONF },
    { "CLK_TCK", _SC_CLK_TCK, SYSCONF },
    { "INT_MAX", _SC_INT_MAX, SYSCONF },
    { "INT_MIN", _SC_INT_MIN, SYSCONF },
    { "IOV_MAX", _SC_UIO_MAXIOV, SYSCONF },
    { "LOGNAME_MAX", _SC_LOGIN_NAME_MAX, SYSCONF },
    { "LONG_BIT", _SC_LONG_BIT, SYSCONF },
    { "MB_LEN_MAX", _SC_MB_LEN_MAX, SYSCONF },
    { "NGROUPS_MAX", _SC_NGROUPS_MAX, SYSCONF },
    { "NL_ARGMAX", _SC_NL_ARGMAX, SYSCONF },
    { "NL_LANGMAX", _SC_NL_LANGMAX, SYSCONF },
    { "NL_MSGMAX", _SC_NL_MSGMAX, SYSCONF },
    { "NL_NMAX", _SC_NL_NMAX, SYSCONF },
    { "NL_SETMAX", _SC_NL_SETMAX, SYSCONF },
    { "NL_TEXTMAX", _SC_NL_TEXTMAX, SYSCONF },
    { "NSS_BUFLEN_GROUP", _SC_GETGR_R_SIZE_MAX, SYSCONF },
    { "NSS_BUFLEN_PASSWD", _SC_GETPW_R_SIZE_MAX, SYSCONF },
    { "NZERO", _SC_NZERO, SYSCONF },
    { "OPEN_MAX", _SC_OPEN_MAX, SYSCONF },
    { "PAGESIZE", _SC_PAGESIZE, SYSCONF },
    { "PAGE_SIZE", _SC_PAGESIZE, SYSCONF },
    { "PASS_MAX", _SC_PASS_MAX, SYSCONF },
    { "PTHREAD_DESTRUCTOR_ITERATIONS", _SC_THREAD_DESTRUCTOR_ITERATIONS, SYSCONF },
    { "PTHREAD_KEYS_MAX", _SC_THREAD_KEYS_MAX, SYSCONF },
    { "PTHREAD_STACK_MIN", _SC_THREAD_STACK_MIN, SYSCONF },
    { "PTHREAD_THREADS_MAX", _SC_THREAD_THREADS_MAX, SYSCONF },
    { "SCHAR_MAX", _SC_SCHAR_MAX, SYSCONF },
    { "SCHAR_MIN", _SC_SCHAR_MIN, SYSCONF },
    { "SHRT_MAX", _SC_SHRT_MAX, SYSCONF },
    { "SHRT_MIN", _SC_SHRT_MIN, SYSCONF },
    { "SSIZE_MAX", _SC_SSIZE_MAX, SYSCONF },
    { "TTY_NAME_MAX", _SC_TTY_NAME_MAX, SYSCONF },
    { "TZNAME_MAX", _SC_TZNAME_MAX, SYSCONF },
    { "UCHAR_MAX", _SC_UCHAR_MAX, SYSCONF },
    { "UINT_MAX", _SC_UINT_MAX, SYSCONF },
    { "UIO_MAXIOV", _SC_UIO_MAXIOV, SYSCONF },
    { "ULONG_MAX", _SC_ULONG_MAX, SYSCONF },
    { "USHRT_MAX", _SC_USHRT_MAX, SYSCONF },
    { "WORD_BIT", _SC_WORD_BIT, SYSCONF },
    { "_AVPHYS_PAGES", _SC_AVPHYS_PAGES, SYSCONF },
    { "_NPROCESSORS_CONF", _SC_NPROCESSORS_CONF, SYSCONF },
    { "_NPROCESSORS_ONLN", _SC_NPROCESSORS_ONLN, SYSCONF },
    { "_PHYS_PAGES", _SC_PHYS_PAGES, SYSCONF },
    { "_POSIX_ARG_MAX", _SC_ARG_MAX, SYSCONF },
    { "_POSIX_ASYNCHRONOUS_IO", _SC_ASYNCHRONOUS_IO, SYSCONF },
    { "_POSIX_CHILD_MAX", _SC_CHILD_MAX, SYSCONF },
    { "_POSIX_FSYNC", _SC_FSYNC, SYSCONF },
    { "_POSIX_JOB_CONTROL", _SC_JOB_CONTROL, SYSCONF },
    { "_POSIX_MAPPED_FILES", _SC_MAPPED_FILES, SYSCONF },
    { "_POSIX_MEMLOCK", _SC_MEMLOCK, SYSCONF },
    { "_POSIX_MEMLOCK_RANGE", _SC_MEMLOCK_RANGE, SYSCONF },
    { "_POSIX_MEMORY_PROTECTION", _SC_MEMORY_PROTECTION, SYSCONF },
    { "_POSIX_MESSAGE_PASSING", _SC_MESSAGE_PASSING, SYSCONF },
    { "_POSIX_NGROUPS_MAX", _SC_NGROUPS_MAX, SYSCONF },
    { "_POSIX_OPEN_MAX", _SC_OPEN_MAX, SYSCONF },
    { "_POSIX_PII", _SC_PII, SYSCONF },
    { "_POSIX_PII_INTERNET", _SC_PII_INTERNET, SYSCONF },
    { "_POSIX_PII_INTERNET_DGRAM", _SC_PII_INTERNET_DGRAM, SYSCONF },
    { "_POSIX_PII_INTERNET_STREAM", _SC_PII_INTERNET_STREAM, SYSCONF },
    { "_POSIX_PII_OSI", _SC_PII_OSI, SYSCONF },
    { "_POSIX_PII_OSI_CLTS", _SC_PII_OSI_CLTS, SYSCONF },
    { "_POSIX_PII_OSI_COTS", _SC_PII_OSI_COTS, SYSCONF },
    { "_POSIX_PII_OSI_M", _SC_PII_OSI_M, SYSCONF },
    { "_POSIX_PII_SOCKET", _SC_PII_SOCKET, SYSCONF },
    { "_POSIX_PII_XTI", _SC_PII_XTI, SYSCONF },
    { "_POSIX_POLL", _SC_POLL, SYSCONF },
    { "_POSIX_PRIORITIZED_IO", _SC_PRIORITIZED_IO, SYSCONF },
    { "_POSIX_PRIORITY_SCHEDULING", _SC_PRIORITY_SCHEDULING, SYSCONF },
    { "_POSIX_REALTIME_SIGNALS", _SC_REALTIME_SIGNALS, SYSCONF },
    { "_POSIX_SAVED_IDS", _SC_SAVED_IDS, SYSCONF },
    { "_POSIX_SELECT", _SC_SELECT, SYSCONF },
    { "_POSIX_SEMAPHORES", _SC_SEMAPHORES, SYSCONF },
    { "_POSIX_SHARED_MEMORY_OBJECTS", _SC_SHARED_MEMORY_OBJECTS, SYSCONF },
    { "_POSIX_SSIZE_MAX", _SC_SSIZE_MAX, SYSCONF },
    { "_POSIX_STREAM_MAX", _SC_STREAM_MAX, SYSCONF },
    { "_POSIX_SYNCHRONIZED_IO", _SC_SYNCHRONIZED_IO, SYSCONF },
    { "_POSIX_THREADS", _SC_THREADS, SYSCONF },
    { "_POSIX_THREAD_ATTR_STACKADDR", _SC_THREAD_ATTR_STACKADDR, SYSCONF },
    { "_POSIX_THREAD_ATTR_STACKSIZE", _SC_THREAD_ATTR_STACKSIZE, SYSCONF },
    { "_POSIX_THREAD_PRIORITY_SCHEDULING", _SC_THREAD_PRIORITY_SCHEDULING, SYSCONF },
    { "_POSIX_THREAD_PRIO_INHERIT", _SC_THREAD_PRIO_INHERIT, SYSCONF },
    { "_POSIX_THREAD_PRIO_PROTECT", _SC_THREAD_PRIO_PROTECT, SYSCONF },
    { "_POSIX_THREAD_ROBUST_PRIO_INHERIT", _SC_THREAD_ROBUST_PRIO_INHERIT,
      SYSCONF },
    { "_POSIX_THREAD_ROBUST_PRIO_PROTECT", _SC_THREAD_ROBUST_PRIO_PROTECT,
      SYSCONF },
    { "_POSIX_THREAD_PROCESS_SHARED", _SC_THREAD_PROCESS_SHARED, SYSCONF },
    { "_POSIX_THREAD_SAFE_FUNCTIONS", _SC_THREAD_SAFE_FUNCTIONS, SYSCONF },
    { "_POSIX_TIMERS", _SC_TIMERS, SYSCONF },
    { "TIMER_MAX", _SC_TIMER_MAX, SYSCONF },
    { "_POSIX_TZNAME_MAX", _SC_TZNAME_MAX, SYSCONF },
    { "_POSIX_VERSION", _SC_VERSION, SYSCONF },
    { "_T_IOV_MAX", _SC_T_IOV_MAX, SYSCONF },
    { "_XOPEN_CRYPT", _SC_XOPEN_CRYPT, SYSCONF },
    { "_XOPEN_ENH_I18N", _SC_XOPEN_ENH_I18N, SYSCONF },
    { "_XOPEN_LEGACY", _SC_XOPEN_LEGACY, SYSCONF },
    { "_XOPEN_REALTIME", _SC_XOPEN_REALTIME, SYSCONF },
    { "_XOPEN_REALTIME_THREADS", _SC_XOPEN_REALTIME_THREADS, SYSCONF },
    { "_XOPEN_SHM", _SC_XOPEN_SHM, SYSCONF },
    { "_XOPEN_UNIX", _SC_XOPEN_UNIX, SYSCONF },
    { "_XOPEN_VERSION", _SC_XOPEN_VERSION, SYSCONF },
    { "_XOPEN_XCU_VERSION", _SC_XOPEN_XCU_VERSION, SYSCONF },
    { "_XOPEN_XPG2", _SC_XOPEN_XPG2, SYSCONF },
    { "_XOPEN_XPG3", _SC_XOPEN_XPG3, SYSCONF },
    { "_XOPEN_XPG4", _SC_XOPEN_XPG4, SYSCONF },
    /* POSIX.2  */
    { "BC_BASE_MAX", _SC_BC_BASE_MAX, SYSCONF },
    { "BC_DIM_MAX", _SC_BC_DIM_MAX, SYSCONF },
    { "BC_SCALE_MAX", _SC_BC_SCALE_MAX, SYSCONF },
    { "BC_STRING_MAX", _SC_BC_STRING_MAX, SYSCONF },
    { "CHARCLASS_NAME_MAX", _SC_CHARCLASS_NAME_MAX, SYSCONF },
    { "COLL_WEIGHTS_MAX", _SC_COLL_WEIGHTS_MAX, SYSCONF },
    { "EQUIV_CLASS_MAX", _SC_EQUIV_CLASS_MAX, SYSCONF },
    { "EXPR_NEST_MAX", _SC_EXPR_NEST_MAX, SYSCONF },
    { "LINE_MAX", _SC_LINE_MAX, SYSCONF },
    { "POSIX2_BC_BASE_MAX", _SC_BC_BASE_MAX, SYSCONF },
    { "POSIX2_BC_DIM_MAX", _SC_BC_DIM_MAX, SYSCONF },
    { "POSIX2_BC_SCALE_MAX", _SC_BC_SCALE_MAX, SYSCONF },
    { "POSIX2_BC_STRING_MAX", _SC_BC_STRING_MAX, SYSCONF },
    { "POSIX2_CHAR_TERM", _SC_2_CHAR_TERM, SYSCONF },
    { "POSIX2_COLL_WEIGHTS_MAX", _SC_COLL_WEIGHTS_MAX, SYSCONF },
    { "POSIX2_C_BIND", _SC_2_C_BIND, SYSCONF },
    { "POSIX2_C_DEV", _SC_2_C_DEV, SYSCONF },
    { "POSIX2_C_VERSION", _SC_2_C_VERSION, SYSCONF },
    { "POSIX2_EXPR_NEST_MAX", _SC_EXPR_NEST_MAX, SYSCONF },
    { "POSIX2_FORT_DEV", _SC_2_FORT_DEV, SYSCONF },
    { "POSIX2_FORT_RUN", _SC_2_FORT_RUN, SYSCONF },
    { "_POSIX2_LINE_MAX", _SC_LINE_MAX, SYSCONF },
    { "POSIX2_LINE_MAX", _SC_LINE_MAX, SYSCONF },
    { "POSIX2_LOCALEDEF", _SC_2_LOCALEDEF, SYSCONF },
    { "POSIX2_RE_DUP_MAX", _SC_RE_DUP_MAX, SYSCONF },
    { "POSIX2_SW_DEV", _SC_2_SW_DEV, SYSCONF },
    { "POSIX2_UPE", _SC_2_UPE, SYSCONF },
    { "POSIX2_VERSION", _SC_2_VERSION, SYSCONF },
    { "RE_DUP_MAX", _SC_RE_DUP_MAX, SYSCONF },

    { "PATH", _CS_PATH, CONFSTR },
    { "CS_PATH", _CS_PATH, CONFSTR },

    /* LFS */
    { "LFS_CFLAGS", _CS_LFS_CFLAGS, CONFSTR },
    { "LFS_LDFLAGS", _CS_LFS_LDFLAGS, CONFSTR },
    { "LFS_LIBS", _CS_LFS_LIBS, CONFSTR },
    { "LFS_LINTFLAGS", _CS_LFS_LINTFLAGS, CONFSTR },
    { "LFS64_CFLAGS", _CS_LFS64_CFLAGS, CONFSTR },
    { "LFS64_LDFLAGS", _CS_LFS64_LDFLAGS, CONFSTR },
    { "LFS64_LIBS", _CS_LFS64_LIBS, CONFSTR },
    { "LFS64_LINTFLAGS", _CS_LFS64_LINTFLAGS, CONFSTR },

    /* Programming environments.  */
    { "_XBS5_WIDTH_RESTRICTED_ENVS", _CS_V5_WIDTH_RESTRICTED_ENVS, CONFSTR },
    { "XBS5_WIDTH_RESTRICTED_ENVS", _CS_V5_WIDTH_RESTRICTED_ENVS, CONFSTR },

    { "_XBS5_ILP32_OFF32", _SC_XBS5_ILP32_OFF32, SYSCONF },
    { "XBS5_ILP32_OFF32_CFLAGS", _CS_XBS5_ILP32_OFF32_CFLAGS, CONFSTR },
    { "XBS5_ILP32_OFF32_LDFLAGS", _CS_XBS5_ILP32_OFF32_LDFLAGS, CONFSTR },
    { "XBS5_ILP32_OFF32_LIBS", _CS_XBS5_ILP32_OFF32_LIBS, CONFSTR },
    { "XBS5_ILP32_OFF32_LINTFLAGS", _CS_XBS5_ILP32_OFF32_LINTFLAGS, CONFSTR },

    { "_XBS5_ILP32_OFFBIG", _SC_XBS5_ILP32_OFFBIG, SYSCONF },
    { "XBS5_ILP32_OFFBIG_CFLAGS", _CS_XBS5_ILP32_OFFBIG_CFLAGS, CONFSTR },
    { "XBS5_ILP32_OFFBIG_LDFLAGS", _CS_XBS5_ILP32_OFFBIG_LDFLAGS, CONFSTR },
    { "XBS5_ILP32_OFFBIG_LIBS", _CS_XBS5_ILP32_OFFBIG_LIBS, CONFSTR },
    { "XBS5_ILP32_OFFBIG_LINTFLAGS", _CS_XBS5_ILP32_OFFBIG_LINTFLAGS, CONFSTR },

    { "_XBS5_LP64_OFF64", _SC_XBS5_LP64_OFF64, SYSCONF },
    { "XBS5_LP64_OFF64_CFLAGS", _CS_XBS5_LP64_OFF64_CFLAGS, CONFSTR },
    { "XBS5_LP64_OFF64_LDFLAGS", _CS_XBS5_LP64_OFF64_LDFLAGS, CONFSTR },
    { "XBS5_LP64_OFF64_LIBS", _CS_XBS5_LP64_OFF64_LIBS, CONFSTR },
    { "XBS5_LP64_OFF64_LINTFLAGS", _CS_XBS5_LP64_OFF64_LINTFLAGS, CONFSTR },

    { "_XBS5_LPBIG_OFFBIG", _SC_XBS5_LPBIG_OFFBIG, SYSCONF },
    { "XBS5_LPBIG_OFFBIG_CFLAGS", _CS_XBS5_LPBIG_OFFBIG_CFLAGS, CONFSTR },
    { "XBS5_LPBIG_OFFBIG_LDFLAGS", _CS_XBS5_LPBIG_OFFBIG_LDFLAGS, CONFSTR },
    { "XBS5_LPBIG_OFFBIG_LIBS", _CS_XBS5_LPBIG_OFFBIG_LIBS, CONFSTR },
    { "XBS5_LPBIG_OFFBIG_LINTFLAGS", _CS_XBS5_LPBIG_OFFBIG_LINTFLAGS, CONFSTR },

    { "_POSIX_V6_ILP32_OFF32", _SC_V6_ILP32_OFF32, SYSCONF },
    { "POSIX_V6_ILP32_OFF32_CFLAGS", _CS_POSIX_V6_ILP32_OFF32_CFLAGS, CONFSTR },
    { "POSIX_V6_ILP32_OFF32_LDFLAGS", _CS_POSIX_V6_ILP32_OFF32_LDFLAGS, CONFSTR },
    { "POSIX_V6_ILP32_OFF32_LIBS", _CS_POSIX_V6_ILP32_OFF32_LIBS, CONFSTR },
    { "POSIX_V6_ILP32_OFF32_LINTFLAGS", _CS_POSIX_V6_ILP32_OFF32_LINTFLAGS, CONFSTR },

    { "_POSIX_V6_WIDTH_RESTRICTED_ENVS", _CS_V6_WIDTH_RESTRICTED_ENVS, CONFSTR },
    { "POSIX_V6_WIDTH_RESTRICTED_ENVS", _CS_V6_WIDTH_RESTRICTED_ENVS, CONFSTR },

    { "_POSIX_V6_ILP32_OFFBIG", _SC_V6_ILP32_OFFBIG, SYSCONF },
    { "POSIX_V6_ILP32_OFFBIG_CFLAGS", _CS_POSIX_V6_ILP32_OFFBIG_CFLAGS, CONFSTR },
    { "POSIX_V6_ILP32_OFFBIG_LDFLAGS", _CS_POSIX_V6_ILP32_OFFBIG_LDFLAGS, CONFSTR },
    { "POSIX_V6_ILP32_OFFBIG_LIBS", _CS_POSIX_V6_ILP32_OFFBIG_LIBS, CONFSTR },
    { "POSIX_V6_ILP32_OFFBIG_LINTFLAGS", _CS_POSIX_V6_ILP32_OFFBIG_LINTFLAGS, CONFSTR },

    { "_POSIX_V6_LP64_OFF64", _SC_V6_LP64_OFF64, SYSCONF },
    { "POSIX_V6_LP64_OFF64_CFLAGS", _CS_POSIX_V6_LP64_OFF64_CFLAGS, CONFSTR },
    { "POSIX_V6_LP64_OFF64_LDFLAGS", _CS_POSIX_V6_LP64_OFF64_LDFLAGS, CONFSTR },
    { "POSIX_V6_LP64_OFF64_LIBS", _CS_POSIX_V6_LP64_OFF64_LIBS, CONFSTR },
    { "POSIX_V6_LP64_OFF64_LINTFLAGS", _CS_POSIX_V6_LP64_OFF64_LINTFLAGS, CONFSTR },

    { "_POSIX_V6_LPBIG_OFFBIG", _SC_V6_LPBIG_OFFBIG, SYSCONF },
    { "POSIX_V6_LPBIG_OFFBIG_CFLAGS", _CS_POSIX_V6_LPBIG_OFFBIG_CFLAGS, CONFSTR },
    { "POSIX_V6_LPBIG_OFFBIG_LDFLAGS", _CS_POSIX_V6_LPBIG_OFFBIG_LDFLAGS, CONFSTR },
    { "POSIX_V6_LPBIG_OFFBIG_LIBS", _CS_POSIX_V6_LPBIG_OFFBIG_LIBS, CONFSTR },
    { "POSIX_V6_LPBIG_OFFBIG_LINTFLAGS", _CS_POSIX_V6_LPBIG_OFFBIG_LINTFLAGS, CONFSTR },

    { "_POSIX_V7_ILP32_OFF32", _SC_V7_ILP32_OFF32, SYSCONF },
    { "POSIX_V7_ILP32_OFF32_CFLAGS", _CS_POSIX_V7_ILP32_OFF32_CFLAGS, CONFSTR },
    { "POSIX_V7_ILP32_OFF32_LDFLAGS", _CS_POSIX_V7_ILP32_OFF32_LDFLAGS, CONFSTR },
    { "POSIX_V7_ILP32_OFF32_LIBS", _CS_POSIX_V7_ILP32_OFF32_LIBS, CONFSTR },
    { "POSIX_V7_ILP32_OFF32_LINTFLAGS", _CS_POSIX_V7_ILP32_OFF32_LINTFLAGS, CONFSTR },

    { "_POSIX_V7_WIDTH_RESTRICTED_ENVS", _CS_V7_WIDTH_RESTRICTED_ENVS, CONFSTR },
    { "POSIX_V7_WIDTH_RESTRICTED_ENVS", _CS_V7_WIDTH_RESTRICTED_ENVS, CONFSTR },

    { "_POSIX_V7_ILP32_OFFBIG", _SC_V7_ILP32_OFFBIG, SYSCONF },
    { "POSIX_V7_ILP32_OFFBIG_CFLAGS", _CS_POSIX_V7_ILP32_OFFBIG_CFLAGS, CONFSTR },
    { "POSIX_V7_ILP32_OFFBIG_LDFLAGS", _CS_POSIX_V7_ILP32_OFFBIG_LDFLAGS, CONFSTR },
    { "POSIX_V7_ILP32_OFFBIG_LIBS", _CS_POSIX_V7_ILP32_OFFBIG_LIBS, CONFSTR },
    { "POSIX_V7_ILP32_OFFBIG_LINTFLAGS", _CS_POSIX_V7_ILP32_OFFBIG_LINTFLAGS, CONFSTR },

    { "_POSIX_V7_LP64_OFF64", _SC_V7_LP64_OFF64, SYSCONF },
    { "POSIX_V7_LP64_OFF64_CFLAGS", _CS_POSIX_V7_LP64_OFF64_CFLAGS, CONFSTR },
    { "POSIX_V7_LP64_OFF64_LDFLAGS", _CS_POSIX_V7_LP64_OFF64_LDFLAGS, CONFSTR },
    { "POSIX_V7_LP64_OFF64_LIBS", _CS_POSIX_V7_LP64_OFF64_LIBS, CONFSTR },
    { "POSIX_V7_LP64_OFF64_LINTFLAGS", _CS_POSIX_V7_LP64_OFF64_LINTFLAGS, CONFSTR },

    { "_POSIX_V7_LPBIG_OFFBIG", _SC_V7_LPBIG_OFFBIG, SYSCONF },
    { "POSIX_V7_LPBIG_OFFBIG_CFLAGS", _CS_POSIX_V7_LPBIG_OFFBIG_CFLAGS, CONFSTR },
    { "POSIX_V7_LPBIG_OFFBIG_LDFLAGS", _CS_POSIX_V7_LPBIG_OFFBIG_LDFLAGS, CONFSTR },
    { "POSIX_V7_LPBIG_OFFBIG_LIBS", _CS_POSIX_V7_LPBIG_OFFBIG_LIBS, CONFSTR },
    { "POSIX_V7_LPBIG_OFFBIG_LINTFLAGS", _CS_POSIX_V7_LPBIG_OFFBIG_LINTFLAGS, CONFSTR },

    { "_POSIX_ADVISORY_INFO", _SC_ADVISORY_INFO, SYSCONF },
    { "_POSIX_BARRIERS", _SC_BARRIERS, SYSCONF },
    { "_POSIX_BASE", _SC_BASE, SYSCONF },
    { "_POSIX_C_LANG_SUPPORT", _SC_C_LANG_SUPPORT, SYSCONF },
    { "_POSIX_C_LANG_SUPPORT_R", _SC_C_LANG_SUPPORT_R, SYSCONF },
    { "_POSIX_CLOCK_SELECTION", _SC_CLOCK_SELECTION, SYSCONF },
    { "_POSIX_CPUTIME", _SC_CPUTIME, SYSCONF },
    { "_POSIX_THREAD_CPUTIME", _SC_THREAD_CPUTIME, SYSCONF },
    { "_POSIX_DEVICE_SPECIFIC", _SC_DEVICE_SPECIFIC, SYSCONF },
    { "_POSIX_DEVICE_SPECIFIC_R", _SC_DEVICE_SPECIFIC_R, SYSCONF },
    { "_POSIX_FD_MGMT", _SC_FD_MGMT, SYSCONF },
    { "_POSIX_FIFO", _SC_FIFO, SYSCONF },
    { "_POSIX_PIPE", _SC_PIPE, SYSCONF },
    { "_POSIX_FILE_ATTRIBUTES", _SC_FILE_ATTRIBUTES, SYSCONF },
    { "_POSIX_FILE_LOCKING", _SC_FILE_LOCKING, SYSCONF },
    { "_POSIX_FILE_SYSTEM", _SC_FILE_SYSTEM, SYSCONF },
    { "_POSIX_MONOTONIC_CLOCK", _SC_MONOTONIC_CLOCK, SYSCONF },
    { "_POSIX_MULTI_PROCESS", _SC_MULTI_PROCESS, SYSCONF },
    { "_POSIX_SINGLE_PROCESS", _SC_SINGLE_PROCESS, SYSCONF },
    { "_POSIX_NETWORKING", _SC_NETWORKING, SYSCONF },
    { "_POSIX_READER_WRITER_LOCKS", _SC_READER_WRITER_LOCKS, SYSCONF },
    { "_POSIX_SPIN_LOCKS", _SC_SPIN_LOCKS, SYSCONF },
    { "_POSIX_REGEXP", _SC_REGEXP, SYSCONF },
    { "_REGEX_VERSION", _SC_REGEX_VERSION, SYSCONF },
    { "_POSIX_SHELL", _SC_SHELL, SYSCONF },
    { "_POSIX_SIGNALS", _SC_SIGNALS, SYSCONF },
    { "_POSIX_SPAWN", _SC_SPAWN, SYSCONF },
    { "_POSIX_SPORADIC_SERVER", _SC_SPORADIC_SERVER, SYSCONF },
    { "_POSIX_THREAD_SPORADIC_SERVER", _SC_THREAD_SPORADIC_SERVER, SYSCONF },
    { "_POSIX_SYSTEM_DATABASE", _SC_SYSTEM_DATABASE, SYSCONF },
    { "_POSIX_SYSTEM_DATABASE_R", _SC_SYSTEM_DATABASE_R, SYSCONF },
    { "_POSIX_TIMEOUTS", _SC_TIMEOUTS, SYSCONF },
    { "_POSIX_TYPED_MEMORY_OBJECTS", _SC_TYPED_MEMORY_OBJECTS, SYSCONF },
    { "_POSIX_USER_GROUPS", _SC_USER_GROUPS, SYSCONF },
    { "_POSIX_USER_GROUPS_R", _SC_USER_GROUPS_R, SYSCONF },
    { "POSIX2_PBS", _SC_2_PBS, SYSCONF },
    { "POSIX2_PBS_ACCOUNTING", _SC_2_PBS_ACCOUNTING, SYSCONF },
    { "POSIX2_PBS_LOCATE", _SC_2_PBS_LOCATE, SYSCONF },
    { "POSIX2_PBS_TRACK", _SC_2_PBS_TRACK, SYSCONF },
    { "POSIX2_PBS_MESSAGE", _SC_2_PBS_MESSAGE, SYSCONF },
    { "SYMLOOP_MAX", _SC_SYMLOOP_MAX, SYSCONF },
    { "STREAM_MAX", _SC_STREAM_MAX, SYSCONF },
    { "AIO_LISTIO_MAX", _SC_AIO_LISTIO_MAX, SYSCONF },
    { "AIO_MAX", _SC_AIO_MAX, SYSCONF },
    { "AIO_PRIO_DELTA_MAX", _SC_AIO_PRIO_DELTA_MAX, SYSCONF },
    { "DELAYTIMER_MAX", _SC_DELAYTIMER_MAX, SYSCONF },
    { "HOST_NAME_MAX", _SC_HOST_NAME_MAX, SYSCONF },
    { "LOGIN_NAME_MAX", _SC_LOGIN_NAME_MAX, SYSCONF },
    { "MQ_OPEN_MAX", _SC_MQ_OPEN_MAX, SYSCONF },
    { "MQ_PRIO_MAX", _SC_MQ_PRIO_MAX, SYSCONF },
    { "_POSIX_DEVICE_IO", _SC_DEVICE_IO, SYSCONF },
    { "_POSIX_TRACE", _SC_TRACE, SYSCONF },
    { "_POSIX_TRACE_EVENT_FILTER", _SC_TRACE_EVENT_FILTER, SYSCONF },
    { "_POSIX_TRACE_INHERIT", _SC_TRACE_INHERIT, SYSCONF },
    { "_POSIX_TRACE_LOG", _SC_TRACE_LOG, SYSCONF },
    { "RTSIG_MAX", _SC_RTSIG_MAX, SYSCONF },
    { "SEM_NSEMS_MAX", _SC_SEM_NSEMS_MAX, SYSCONF },
    { "SEM_VALUE_MAX", _SC_SEM_VALUE_MAX, SYSCONF },
    { "SIGQUEUE_MAX", _SC_SIGQUEUE_MAX, SYSCONF },
    { "FILESIZEBITS", _PC_FILESIZEBITS, PATHCONF },
    { "POSIX_ALLOC_SIZE_MIN", _PC_ALLOC_SIZE_MIN, PATHCONF },
    { "POSIX_REC_INCR_XFER_SIZE", _PC_REC_INCR_XFER_SIZE, PATHCONF },
    { "POSIX_REC_MAX_XFER_SIZE", _PC_REC_MAX_XFER_SIZE, PATHCONF },
    { "POSIX_REC_MIN_XFER_SIZE", _PC_REC_MIN_XFER_SIZE, PATHCONF },
    { "POSIX_REC_XFER_ALIGN", _PC_REC_XFER_ALIGN, PATHCONF },
    { "SYMLINK_MAX", _PC_SYMLINK_MAX, PATHCONF },
    { "GNU_LIBC_VERSION", _CS_GNU_LIBC_VERSION, CONFSTR },
    { "GNU_LIBPTHREAD_VERSION", _CS_GNU_LIBPTHREAD_VERSION, CONFSTR },
    { "POSIX2_SYMLINKS", _PC_2_SYMLINKS, PATHCONF },

    { "LEVEL1_ICACHE_SIZE", _SC_LEVEL1_ICACHE_SIZE, SYSCONF },
    { "LEVEL1_ICACHE_ASSOC", _SC_LEVEL1_ICACHE_ASSOC, SYSCONF },
    { "LEVEL1_ICACHE_LINESIZE", _SC_LEVEL1_ICACHE_LINESIZE, SYSCONF },
    { "LEVEL1_DCACHE_SIZE", _SC_LEVEL1_DCACHE_SIZE, SYSCONF },
    { "LEVEL1_DCACHE_ASSOC", _SC_LEVEL1_DCACHE_ASSOC, SYSCONF },
    { "LEVEL1_DCACHE_LINESIZE", _SC_LEVEL1_DCACHE_LINESIZE, SYSCONF },
    { "LEVEL2_CACHE_SIZE", _SC_LEVEL2_CACHE_SIZE, SYSCONF },
    { "LEVEL2_CACHE_ASSOC", _SC_LEVEL2_CACHE_ASSOC, SYSCONF },
    { "LEVEL2_CACHE_LINESIZE", _SC_LEVEL2_CACHE_LINESIZE, SYSCONF },
    { "LEVEL3_CACHE_SIZE", _SC_LEVEL3_CACHE_SIZE, SYSCONF },
    { "LEVEL3_CACHE_ASSOC", _SC_LEVEL3_CACHE_ASSOC, SYSCONF },
    { "LEVEL3_CACHE_LINESIZE", _SC_LEVEL3_CACHE_LINESIZE, SYSCONF },
    { "LEVEL4_CACHE_SIZE", _SC_LEVEL4_CACHE_SIZE, SYSCONF },
    { "LEVEL4_CACHE_ASSOC", _SC_LEVEL4_CACHE_ASSOC, SYSCONF },
    { "LEVEL4_CACHE_LINESIZE", _SC_LEVEL4_CACHE_LINESIZE, SYSCONF },

    { "IPV6", _SC_IPV6, SYSCONF },
    { "RAW_SOCKETS", _SC_RAW_SOCKETS, SYSCONF },

    { "_POSIX_IPV6", _SC_IPV6, SYSCONF },
    { "_POSIX_RAW_SOCKETS", _SC_RAW_SOCKETS, SYSCONF },

    { NULL, 0, SYSCONF }
  };


extern const char *__progname;


static void
usage (void)
{
  fprintf (stderr,
	   _("Usage: %s [-v specification] variable_name [pathname]\n"),
	   __progname);
  fprintf (stderr,
	   _("       %s -a [pathname]\n"), __progname);
  exit (2);
}


static void
print_all (const char *path)
{
  const struct conf *c;
  size_t clen;
  long int value;
  char *cvalue;
  for (c = vars; c->name != NULL; ++c) {
    printf("%-35s", c->name);
    switch (c->call) {
      case PATHCONF:
	value = pathconf (path, c->call_name);
	if (value != -1) {
	  printf("%ld", value);
	}
	printf("\n");
	break;
      case SYSCONF:
	value = sysconf (c->call_name);
	if (value == -1l) {
	  if (c->call_name == _SC_UINT_MAX
	    || c->call_name == _SC_ULONG_MAX)
	    printf ("%lu", value);
	}
	else {
	  printf ("%ld", value);
	}
	printf ("\n");
	break;
      case CONFSTR:
	clen = confstr (c->call_name, (char *) NULL, 0);
	cvalue = (char *) malloc (clen);
	if (cvalue == NULL)
	  error (3, 0, _("memory exhausted"));
	if (confstr (c->call_name, cvalue, clen) != clen)
	  error (3, errno, "confstr");
	printf ("%.*s\n", (int) clen, cvalue);
	free (cvalue);
	break;
    }
  }
  exit (0);
}

int
main (int argc, char *argv[])
{
  const struct conf *c;

  /* Set locale.  Do not set LC_ALL because the other categories must
     not be affected (according to POSIX.2).  */
  setlocale (LC_CTYPE, "");
  setlocale (LC_MESSAGES, "");

  /* Initialize the message catalog.  */
  textdomain (PACKAGE);

  if (argc > 1 && strcmp (argv[1], "--version") == 0)
    {
      printf ("getconf %s%s\n", PKGVERSION, VERSION);
      printf (gettext ("\
Copyright (C) %s Free Software Foundation, Inc.\n\
This is free software; see the source for copying conditions.  There is NO\n\
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\
"), "2021");
      printf (gettext ("Written by %s.\n"), "Roland McGrath");
      return 0;
    }

  if (argc > 1 && strcmp (argv[1], "--help") == 0)
    {
      printf (gettext ("\
Usage: getconf [-v SPEC] VAR\n\
  or:  getconf [-v SPEC] PATH_VAR PATH\n\
\n\
Get the configuration value for variable VAR, or for variable PATH_VAR\n\
for path PATH.  If SPEC is given, give values for compilation\n\
environment SPEC.\n\n"));
      printf (gettext ("For bug reporting instructions, please see:\n\
%s.\n"), REPORT_BUGS_TO);
      return 0;
    }

#ifdef ALL_ENVIRONMENTS_DEFINED
  if (argc > 1 && strncmp (argv[1], "-v", 2) == 0)
    {
      if (argv[1][2] == '\0')
	{
	  if (argc < 3)
	    usage ();

	  argv += 2;
	  argc -= 2;
	}
      else
	{
	  argv += 1;
	  argc -= 1;
	}
    }
#else
  const char *getconf_dir = getenv ("GETCONF_DIR") ?: GETCONF_DIR;
  size_t getconf_dirlen = strlen (getconf_dir);

  const char *spec = NULL;
  char buf[sizeof "POSIX_V6_LPBIG_OFFBIG"];
  char *argv0 = argv[0];
  if (argc > 1 && strncmp (argv[1], "-v", 2) == 0)
    {
      if (argv[1][2] == '\0')
	{
	  if (argc < 3)
	    usage ();

	  spec = argv[2];
	  argv += 2;
	  argc -= 2;
	}
      else
	{
	  spec = &argv[1][2];
	  argv += 1;
	  argc -= 1;
	}
    }
  else
    {
      char default_name[getconf_dirlen + sizeof "/default"];
      memcpy (mempcpy (default_name, getconf_dir, getconf_dirlen),
	      "/default", sizeof "/default");
      int len = readlink (default_name, buf, sizeof buf - 1);
      if (len > 0)
	{
	  buf[len] = '\0';
	  spec = buf;
	}
    }

  /* Check for the specifications we know.  */
  if (spec != NULL)
    {
      size_t i;
      for (i = 0; i < nspecs; ++i)
	if (strcmp (spec, specs[i].name) == 0)
	  break;

      if (i == nspecs)
	error (2, 0, _("unknown specification \"%s\""), spec);

      switch (specs[i].num)
	{
# ifndef _XBS5_ILP32_OFF32
	  case _SC_XBS5_ILP32_OFF32:
# endif
# ifndef _XBS5_ILP32_OFFBIG
	  case _SC_XBS5_ILP32_OFFBIG:
# endif
# ifndef _XBS5_LP64_OFF64
	  case _SC_XBS5_LP64_OFF64:
# endif
# ifndef _XBS5_LPBIG_OFFBIG
	  case _SC_XBS5_LPBIG_OFFBIG:
# endif
# ifndef _POSIX_V6_ILP32_OFF32
	  case _SC_V6_ILP32_OFF32:
# endif
# ifndef _POSIX_V6_ILP32_OFFBIG
	  case _SC_V6_ILP32_OFFBIG:
# endif
# ifndef _POSIX_V6_LP64_OFF64
	  case _SC_V6_LP64_OFF64:
# endif
# ifndef _POSIX_V6_LPBIG_OFFBIG
	  case _SC_V6_LPBIG_OFFBIG:
# endif
# ifndef _POSIX_V7_ILP32_OFF32
	  case _SC_V7_ILP32_OFF32:
# endif
# ifndef _POSIX_V7_ILP32_OFFBIG
	  case _SC_V7_ILP32_OFFBIG:
# endif
# ifndef _POSIX_V7_LP64_OFF64
	  case _SC_V7_LP64_OFF64:
# endif
# ifndef _POSIX_V7_LPBIG_OFFBIG
	  case _SC_V7_LPBIG_OFFBIG:
# endif
	    {
	      const char *args[argc + 3];
	      size_t spec_len = strlen (spec);
	      char getconf_name[getconf_dirlen + 1 + spec_len + 1];
	      memcpy (mempcpy (mempcpy (getconf_name, getconf_dir,
					getconf_dirlen),
			       "/", 1), spec, spec_len + 1);
	      args[0] = argv0;
	      args[1] = "-v";
	      args[2] = spec;
	      memcpy (&args[3], &argv[1], argc * sizeof (argv[1]));
	      execv (getconf_name, (char * const *) args);
	      error (4, errno, _("Couldn't execute %s"), getconf_name);
	    }
	  default:
	    break;
	}
    }
#endif

  if (argc > 1 && strcmp (argv[1], "-a") == 0)
    {
      if (argc == 2)
	print_all ("/");
      else if (argc == 3)
	print_all (argv[2]);
      else
	usage ();
    }

  int ai = 1;
  if (argc > ai && strcmp (argv[ai], "--") == 0)
    ++ai;

  if (argc - ai < 1 || argc - ai > 2)
    usage ();

  for (c = vars; c->name != NULL; ++c)
    if (strcmp (c->name, argv[ai]) == 0
	|| (strncmp (c->name, "_POSIX_", 7) == 0
	    && strcmp (c->name + 7, argv[ai]) == 0))
      {
	long int value;
	size_t clen;
	char *cvalue;
	switch (c->call)
	  {
	  case PATHCONF:
	    if (argc - ai < 2)
	      usage ();
	    errno = 0;
	    value = pathconf (argv[ai + 1], c->call_name);
	    if (value == -1)
	      {
		if (errno)
		  error (3, errno, "pathconf: %s", argv[ai + 1]);
		else
		  puts (_("undefined"));
	      }
	    else
	      printf ("%ld\n", value);
	    exit (0);

	  case SYSCONF:
	    if (argc - ai > 1)
	      usage ();
	    value = sysconf (c->call_name);
	    if (value == -1l)
	      {
		if (c->call_name == _SC_UINT_MAX
		    || c->call_name == _SC_ULONG_MAX)
		  printf ("%lu\n", value);
		else
		  puts (_("undefined"));
	      }
	    else
	      printf ("%ld\n", value);
	    exit (0);

	  case CONFSTR:
	    if (argc - ai > 1)
	      usage ();
	    clen = confstr (c->call_name, (char *) NULL, 0);
	    cvalue = (char *) malloc (clen);
	    if (cvalue == NULL)
	      error (3, 0, _("memory exhausted"));

	    if (confstr (c->call_name, cvalue, clen) != clen)
	      error (3, errno, "confstr");

	    printf ("%.*s\n", (int) clen, cvalue);
	    exit (0);
	  }
      }

  error (2, 0, _("Unrecognized variable `%s'"), argv[ai]);
  /* NOTREACHED */
  return 2;
}
