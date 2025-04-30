#ifndef _SYS_RESOURCE_H
#include <resource/sys/resource.h>

#ifndef _ISOMAC
# include <time.h>
# include <string.h>

/* Internal version of rusage with a 64-bit time_t. */
#if __TIMESIZE == 64
# define __rusage64 rusage
#else
struct __rusage64
  {
    struct __timeval64 ru_utime;
    struct __timeval64 ru_stime;
    __extension__ union
      {
	long int ru_maxrss;
	__syscall_slong_t __ru_maxrss_word;
      };
    __extension__ union
      {
	long int ru_ixrss;
	__syscall_slong_t __ru_ixrss_word;
      };
    __extension__ union
      {
	long int ru_idrss;
	__syscall_slong_t __ru_idrss_word;
      };
    __extension__ union
      {
	long int ru_isrss;
	 __syscall_slong_t __ru_isrss_word;
      };
    __extension__ union
      {
	long int ru_minflt;
	__syscall_slong_t __ru_minflt_word;
      };
    __extension__ union
      {
	long int ru_majflt;
	__syscall_slong_t __ru_majflt_word;
      };
    __extension__ union
      {
	long int ru_nswap;
	__syscall_slong_t __ru_nswap_word;
      };
    __extension__ union
      {
	long int ru_inblock;
	__syscall_slong_t __ru_inblock_word;
      };
    __extension__ union
      {
	long int ru_oublock;
	__syscall_slong_t __ru_oublock_word;
      };
    __extension__ union
      {
	long int ru_msgsnd;
	__syscall_slong_t __ru_msgsnd_word;
      };
    __extension__ union
      {
	long int ru_msgrcv;
	__syscall_slong_t __ru_msgrcv_word;
      };
    __extension__ union
      {
	long int ru_nsignals;
	__syscall_slong_t __ru_nsignals_word;
      };
    __extension__ union
      {
	long int ru_nvcsw;
	__syscall_slong_t __ru_nvcsw_word;
      };
    __extension__ union
      {
	long int ru_nivcsw;
	__syscall_slong_t __ru_nivcsw_word;
      };
  };
#endif

static inline void
rusage64_to_rusage (const struct __rusage64 *restrict r64,
                    struct rusage *restrict r)
{
  /* Make sure the entire output structure is cleared, including
     padding and reserved fields.  */
  memset (r, 0, sizeof *r);

  r->ru_utime    = valid_timeval64_to_timeval (r64->ru_utime);
  r->ru_stime    = valid_timeval64_to_timeval (r64->ru_stime);
  r->ru_maxrss   = r64->ru_maxrss;
  r->ru_ixrss    = r64->ru_ixrss;
  r->ru_idrss    = r64->ru_idrss;
  r->ru_isrss    = r64->ru_isrss;
  r->ru_minflt   = r64->ru_minflt;
  r->ru_majflt   = r64->ru_majflt;
  r->ru_nswap    = r64->ru_nswap;
  r->ru_inblock  = r64->ru_inblock;
  r->ru_oublock  = r64->ru_oublock;
  r->ru_msgsnd   = r64->ru_msgsnd;
  r->ru_msgrcv   = r64->ru_msgrcv;
  r->ru_nsignals = r64->ru_nsignals;
  r->ru_nvcsw    = r64->ru_nvcsw;
  r->ru_nivcsw   = r64->ru_nivcsw;
}

/* Prototypes repeated instead of using __typeof because
   sys/resource.h is included in C++ tests, and declaring functions
   with __typeof and __THROW doesn't work for C++.  */
extern int __getpriority (__priority_which_t __which, id_t __who) __THROW;
libc_hidden_proto (__getpriority)
extern int __setpriority (__priority_which_t __which, id_t __who, int __prio)
     __THROW;
libc_hidden_proto (__setpriority)
libc_hidden_proto (getrlimit64)
extern __typeof (getrlimit64) __getrlimit64;
libc_hidden_proto (__getrlimit64);

/* Now define the internal interfaces.  */
extern int __getrlimit (enum __rlimit_resource __resource,
			struct rlimit *__rlimits) __nonnull ((2));
libc_hidden_proto (__getrlimit)
extern int __getrusage (enum __rusage_who __who, struct rusage *__usage)
	attribute_hidden;

extern int __setrlimit (enum __rlimit_resource __resource,
			const struct rlimit *__rlimits) __nonnull ((2));
libc_hidden_proto (__setrlimit);

#if __TIMESIZE == 64
# define __getrusage64 __getrusage
# define __wait4_time64 __wait4
# define __wait3_time64 __wait3
#else
extern int __getrusage64 (enum __rusage_who who, struct __rusage64 *usage);
libc_hidden_proto (__getrusage64)
extern pid_t __wait4_time64 (pid_t pid, int *stat_loc, int options,
                             struct __rusage64 *usage);
libc_hidden_proto (__wait4_time64)
extern pid_t __wait3_time64 (int *stat_loc, int options,
                             struct __rusage64 *usage);
libc_hidden_proto (__wait3_time64)
#endif
#endif
#endif
