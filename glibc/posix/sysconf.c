/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <errno.h>
#include <grp.h>
#include <pwd.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <limits.h>
#include <sys/param.h>
#include <sys/sysinfo.h>


/* Get the value of the system variable NAME.  */
long int
__sysconf (int name)
{
  switch (name)
    {
    default:
      __set_errno (EINVAL);
      return -1;

    case _SC_TZNAME_MAX:
      return -1;

    case _SC_CHARCLASS_NAME_MAX:
#ifdef	CHARCLASS_NAME_MAX
      return CHARCLASS_NAME_MAX;
#else
      return -1;
#endif

    case _SC_COLL_WEIGHTS_MAX:
#ifdef	COLL_WEIGHTS_MAX
      return COLL_WEIGHTS_MAX;
#else
      return -1;
#endif

    case _SC_EQUIV_CLASS_MAX:
#ifdef	EQUIV_CLASS_MAX
      return EQUIV_CLASS_MAX;
#else
      return -1;
#endif

    case _SC_2_LOCALEDEF:
#ifdef	_POSIX2_LOCALEDEF
      return _POSIX2_LOCALEDEF;
#else
      return -1;
#endif

    case _SC_NPROCESSORS_CONF:
      return __get_nprocs_conf ();

    case _SC_NPROCESSORS_ONLN:
      return __get_nprocs ();

    case _SC_PHYS_PAGES:
      return __get_phys_pages ();

    case _SC_AVPHYS_PAGES:
      return __get_avphys_pages ();

    case _SC_ATEXIT_MAX:
      /* We have no limit since we use lists.  */
      return INT_MAX;

    case _SC_PASS_MAX:
      /* We have no limit but since the return value might be used to
	 allocate a buffer we restrict the value.  */
      return BUFSIZ;

    case _SC_CHAR_BIT:
      return CHAR_BIT;

    case _SC_CHAR_MAX:
      return CHAR_MAX;

    case _SC_CHAR_MIN:
      return CHAR_MIN;

    case _SC_INT_MAX:
      return INT_MAX;

    case _SC_INT_MIN:
      return INT_MIN;

    case _SC_LONG_BIT:
      return sizeof (long int) * CHAR_BIT;

    case _SC_WORD_BIT:
      return sizeof (int) * CHAR_BIT;

    case _SC_MB_LEN_MAX:
      return MB_LEN_MAX;

    case _SC_NZERO:
      return NZERO;

    case _SC_SSIZE_MAX:
      return _POSIX_SSIZE_MAX;

    case _SC_SCHAR_MAX:
      return SCHAR_MAX;

    case _SC_SCHAR_MIN:
      return SCHAR_MIN;

    case _SC_SHRT_MAX:
      return SHRT_MAX;

    case _SC_SHRT_MIN:
      return SHRT_MIN;

    case _SC_UCHAR_MAX:
      return UCHAR_MAX;

    case _SC_UINT_MAX:
      return UINT_MAX;

    case _SC_ULONG_MAX:
      return ULONG_MAX;

    case _SC_USHRT_MAX:
      return USHRT_MAX;

    case _SC_GETGR_R_SIZE_MAX:
      return NSS_BUFLEN_GROUP;

    case _SC_GETPW_R_SIZE_MAX:
      return NSS_BUFLEN_PASSWD;

    case _SC_ARG_MAX:
    case _SC_CHILD_MAX:
    case _SC_CLK_TCK:
    case _SC_NGROUPS_MAX:
    case _SC_OPEN_MAX:
    case _SC_STREAM_MAX:
    case _SC_JOB_CONTROL:
    case _SC_SAVED_IDS:
    case _SC_REALTIME_SIGNALS:
    case _SC_PRIORITY_SCHEDULING:
    case _SC_TIMERS:
    case _SC_ASYNCHRONOUS_IO:
    case _SC_PRIORITIZED_IO:
    case _SC_SYNCHRONIZED_IO:
    case _SC_FSYNC:
    case _SC_MAPPED_FILES:
    case _SC_MEMLOCK:
    case _SC_MEMLOCK_RANGE:
    case _SC_MEMORY_PROTECTION:
    case _SC_MESSAGE_PASSING:
    case _SC_SEMAPHORES:
    case _SC_SHARED_MEMORY_OBJECTS:

    case _SC_AIO_LISTIO_MAX:
    case _SC_AIO_MAX:
    case _SC_AIO_PRIO_DELTA_MAX:
    case _SC_DELAYTIMER_MAX:
    case _SC_MQ_OPEN_MAX:
    case _SC_MQ_PRIO_MAX:
    case _SC_VERSION:
    case _SC_PAGESIZE:
    case _SC_RTSIG_MAX:
    case _SC_SEM_NSEMS_MAX:
    case _SC_SEM_VALUE_MAX:
    case _SC_SIGQUEUE_MAX:
    case _SC_TIMER_MAX:

    case _SC_PII:
    case _SC_PII_XTI:
    case _SC_PII_SOCKET:
    case _SC_PII_OSI:
    case _SC_POLL:
    case _SC_SELECT:
    case _SC_UIO_MAXIOV:
    case _SC_PII_INTERNET_STREAM:
    case _SC_PII_INTERNET_DGRAM:
    case _SC_PII_OSI_COTS:
    case _SC_PII_OSI_CLTS:
    case _SC_PII_OSI_M:
    case _SC_T_IOV_MAX:

    case _SC_BC_BASE_MAX:
    case _SC_BC_DIM_MAX:
    case _SC_BC_SCALE_MAX:
    case _SC_BC_STRING_MAX:
    case _SC_EXPR_NEST_MAX:
    case _SC_LINE_MAX:
    case _SC_RE_DUP_MAX:
    case _SC_2_VERSION:
    case _SC_2_C_BIND:
    case _SC_2_C_DEV:
    case _SC_2_FORT_DEV:
    case _SC_2_SW_DEV:
    case _SC_2_CHAR_TERM:
    case _SC_2_C_VERSION:
    case _SC_2_UPE:

    case _SC_THREADS:
    case _SC_THREAD_SAFE_FUNCTIONS:
    case _SC_LOGIN_NAME_MAX:
    case _SC_TTY_NAME_MAX:
    case _SC_THREAD_DESTRUCTOR_ITERATIONS:
    case _SC_THREAD_KEYS_MAX:
    case _SC_THREAD_STACK_MIN:
    case _SC_THREAD_THREADS_MAX:
    case _SC_THREAD_ATTR_STACKADDR:
    case _SC_THREAD_ATTR_STACKSIZE:
    case _SC_THREAD_PRIORITY_SCHEDULING:
    case _SC_THREAD_PRIO_INHERIT:
    case _SC_THREAD_PRIO_PROTECT:
    case _SC_THREAD_PROCESS_SHARED:

    case _SC_XOPEN_VERSION:
    case _SC_XOPEN_XCU_VERSION:
    case _SC_XOPEN_UNIX:
    case _SC_XOPEN_CRYPT:
    case _SC_XOPEN_ENH_I18N:
    case _SC_XOPEN_SHM:
    case _SC_XOPEN_XPG2:
    case _SC_XOPEN_XPG3:
    case _SC_XOPEN_XPG4:

    case _SC_NL_ARGMAX:
    case _SC_NL_LANGMAX:
    case _SC_NL_MSGMAX:
    case _SC_NL_NMAX:
    case _SC_NL_SETMAX:
    case _SC_NL_TEXTMAX:

    case _SC_XBS5_ILP32_OFF32:
    case _SC_XBS5_ILP32_OFFBIG:
    case _SC_XBS5_LP64_OFF64:
    case _SC_XBS5_LPBIG_OFFBIG:

    case _SC_POSIX_V6_ILP32_OFF32:
    case _SC_POSIX_V6_ILP32_OFFBIG:
    case _SC_POSIX_V6_LP64_OFF64:
    case _SC_POSIX_V6_LPBIG_OFFBIG:

    case _SC_POSIX_V7_ILP32_OFF32:
    case _SC_POSIX_V7_ILP32_OFFBIG:
    case _SC_POSIX_V7_LP64_OFF64:
    case _SC_POSIX_V7_LPBIG_OFFBIG:

    case _SC_XOPEN_LEGACY:
    case _SC_XOPEN_REALTIME:
    case _SC_XOPEN_REALTIME_THREADS:

    case _SC_MINSIGSTKSZ:
    case _SC_SIGSTKSZ:

      break;
    }

  __set_errno (ENOSYS);
  return -1;
}

weak_alias (__sysconf, sysconf)
libc_hidden_def (__sysconf)

stub_warning (sysconf)
