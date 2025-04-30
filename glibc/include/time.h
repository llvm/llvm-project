#ifndef _TIME_H
#include <time/time.h>

#ifndef _ISOMAC
# include <bits/types/struct_timeval.h>
# include <struct___timespec64.h>
# include <struct___timeval64.h>
# include <bits/types/locale_t.h>
# include <stdbool.h>
# include <time/mktime-internal.h>
# include <sys/time.h>
# include <time-clockid.h>
# include <sys/time.h>

extern __typeof (strftime_l) __strftime_l;
libc_hidden_proto (__strftime_l)
extern __typeof (strptime_l) __strptime_l;

libc_hidden_proto (asctime)
libc_hidden_proto (mktime)
libc_hidden_proto (timelocal)
libc_hidden_proto (localtime)
libc_hidden_proto (strftime)
libc_hidden_proto (strptime)

extern __typeof (clock_gettime) __clock_gettime;
libc_hidden_proto (__clock_gettime)
extern __typeof (clock_settime) __clock_settime;
libc_hidden_proto (__clock_settime)

extern __typeof (clock_getres) __clock_getres;
libc_hidden_proto (__clock_getres)

extern __typeof (clock_nanosleep) __clock_nanosleep;
libc_hidden_proto (__clock_nanosleep);

#ifdef __linux__
extern __typeof (clock_adjtime) __clock_adjtime;
libc_hidden_proto (__clock_adjtime);
#endif

/* Now define the internal interfaces.  */
struct tm;

/* Defined in mktime.c.  */
extern const unsigned short int __mon_yday[2][13] attribute_hidden;

/* Defined in localtime.c.  */
extern struct tm _tmbuf attribute_hidden;

/* Defined in tzset.c.  */
extern char *__tzstring (const char *string) attribute_hidden;

extern int __use_tzfile attribute_hidden;

extern void __tzfile_read (const char *file, size_t extra,
			   char **extrap) attribute_hidden;
extern void __tzfile_compute (__time64_t timer, int use_localtime,
			      long int *leap_correct, int *leap_hit,
			      struct tm *tp) attribute_hidden;
extern void __tzfile_default (const char *std, const char *dst,
			      int stdoff, int dstoff)
  attribute_hidden;
extern void __tzset_parse_tz (const char *tz) attribute_hidden;
extern void __tz_compute (__time64_t timer, struct tm *tm, int use_localtime)
  __THROW attribute_hidden;


#if __TIMESIZE == 64
# define __itimerspec64 itimerspec
#else
/* The glibc's internal representation of the struct itimerspec.  */
struct __itimerspec64
{
  struct __timespec64 it_interval;
  struct __timespec64 it_value;
};
#endif

#if __TIMESIZE == 64
# define __utimbuf64 utimbuf
# define __itimerval64 itimerval
#else
/* The glibc Y2038-proof struct __utimbuf64 structure for file's access
   and modification time values.  */
struct __utimbuf64
{
  __time64_t actime;		/* Access time.  */
  __time64_t modtime;		/* Modification time.  */
};
/* The glibc's internal representation of the struct itimerval.  */
struct __itimerval64
{
  struct __timeval64 it_interval;
  struct __timeval64 it_value;
};
#endif

#if __TIMESIZE == 64
# define __getitimer64 __getitimer
# define __setitimer64 __setitimer
#else
extern int __getitimer64 (enum __itimer_which __which,
                          struct __itimerval64 *__value);

libc_hidden_proto (__getitimer64)
extern int __setitimer64 (enum __itimer_which __which,
                          const struct __itimerval64 *__restrict __new,
                          struct __itimerval64 *__restrict __old);
libc_hidden_proto (__setitimer64)
#endif

#if __TIMESIZE == 64
# define __ctime64 ctime
#else
extern char *__ctime64 (const __time64_t *__timer) __THROW;
libc_hidden_proto (__ctime64)
#endif

#if __TIMESIZE == 64
# define __ctime64_r ctime_r
#else
extern char *__ctime64_r (const __time64_t *__restrict __timer,
		          char *__restrict __buf) __THROW;
libc_hidden_proto (__ctime64_r)
#endif

#if __TIMESIZE == 64
# define __localtime64 localtime
#else
extern struct tm *__localtime64 (const __time64_t *__timer);
libc_hidden_proto (__localtime64)
#endif

extern struct tm *__localtime_r (const time_t *__timer,
				 struct tm *__tp) attribute_hidden;
#if __TIMESIZE != 64
extern struct tm *__localtime64_r (const __time64_t *__timer,
				   struct tm *__tp);
libc_hidden_proto (__localtime64_r)

extern __time64_t __mktime64 (struct tm *__tp) __THROW;
libc_hidden_proto (__mktime64)
#endif

extern struct tm *__gmtime_r (const time_t *__restrict __timer,
			      struct tm *__restrict __tp);
libc_hidden_proto (__gmtime_r)

#if __TIMESIZE == 64
# define __gmtime64 gmtime
#else
extern struct tm *__gmtime64 (const __time64_t *__timer);
libc_hidden_proto (__gmtime64)

extern struct tm *__gmtime64_r (const __time64_t *__restrict __timer,
				struct tm *__restrict __tp);
libc_hidden_proto (__gmtime64_r)

extern __time64_t __timegm64 (struct tm *__tp) __THROW;
libc_hidden_proto (__timegm64)
#endif

#if __TIMESIZE == 64
# define __clock_settime64 __clock_settime
#else
extern int __clock_settime64 (clockid_t clock_id,
                              const struct __timespec64 *tp);
libc_hidden_proto (__clock_settime64)
#endif

#if __TIMESIZE == 64
# define __clock_getres64 __clock_getres
#else
extern int __clock_getres64 (clockid_t clock_id,
                             struct __timespec64 *tp);
libc_hidden_proto (__clock_getres64);
#endif

#if __TIMESIZE == 64
# define __utime64 __utime
# define __utimes64 __utimes
# define __utimensat64 __utimensat
#else
extern int __utime64 (const char *file, const struct __utimbuf64 *times);
libc_hidden_proto (__utime64)
extern int __utimes64 (const char *file, const struct __timeval64 tvp[2]);
libc_hidden_proto (__utimes64)
extern int __utimensat64 (int fd, const char *file,
                          const struct __timespec64 tsp[2], int flags);
libc_hidden_proto (__utimensat64);
#endif

extern int __utimensat64_helper (int fd, const char *file,
                                 const struct __timespec64 tsp[2], int flags);
libc_hidden_proto (__utimensat64_helper);

#if __TIMESIZE == 64
# define __futimes64 __futimes
# define __futimesat64 __futimesat
# define __lutimes64 __lutimes
# define __futimens64 __futimens
#else
extern int __futimes64 (int fd, const struct __timeval64 tvp64[2]);
libc_hidden_proto (__futimes64);
extern int __futimesat64 (int fd, const char *file,
                          const struct __timeval64 tvp[2]);
libc_hidden_proto (__futimesat64);
extern int __lutimes64 (const char *file, const struct __timeval64 tvp64[2]);
libc_hidden_proto (__lutimes64);
extern int __futimens64 (int fd, const struct __timespec64 tsp[2]);
libc_hidden_proto (__futimens64);
#endif

#if __TIMESIZE == 64
# define __timer_gettime64 __timer_gettime
# define __timerfd_gettime64 __timerfd_gettime
#else
extern int __timer_gettime64 (timer_t timerid, struct __itimerspec64 *value);
extern int __timerfd_gettime64 (int fd, struct __itimerspec64 *value);
# if PTHREAD_IN_LIBC
libc_hidden_proto (__timer_gettime64)
# else
librt_hidden_proto (__timer_gettime64)
# endif
libc_hidden_proto (__timerfd_gettime64);
#endif

#if __TIMESIZE == 64
# define __timer_settime64 __timer_settime
# define __timerfd_settime64 __timerfd_settime
#else
extern int __timer_settime64 (timer_t timerid, int flags,
                              const struct __itimerspec64 *value,
                              struct __itimerspec64 *ovalue);
extern int __timerfd_settime64 (int fd, int flags,
                                const struct __itimerspec64 *value,
                                struct __itimerspec64 *ovalue);
# if PTHREAD_IN_LIBC
libc_hidden_proto (__timer_settime64)
#else
librt_hidden_proto (__timer_settime64)
#endif
libc_hidden_proto (__timerfd_settime64);
#endif

#if __TIMESIZE == 64
# define __sched_rr_get_interval64 __sched_rr_get_interval
#else
extern int __sched_rr_get_interval64 (pid_t pid, struct __timespec64 *tp);
libc_hidden_proto (__sched_rr_get_interval64);
#endif

#if __TIMESIZE == 64
# define __settimeofday64 __settimeofday
# define __gettimeofday64 __gettimeofday
#else
extern int __settimeofday64 (const struct __timeval64 *tv,
                             const struct timezone *tz);
libc_hidden_proto (__settimeofday64)
extern int __gettimeofday64 (struct __timeval64 *restrict tv,
                             void *restrict tz);
libc_hidden_proto (__gettimeofday64)
#endif

/* Compute the `struct tm' representation of T,
   offset OFFSET seconds east of UTC,
   and store year, yday, mon, mday, wday, hour, min, sec into *TP.
   Return nonzero if successful.  */
extern int __offtime (__time64_t __timer,
		      long int __offset,
		      struct tm *__tp) attribute_hidden;

extern char *__asctime_r (const struct tm *__tp, char *__buf)
  attribute_hidden;
extern void __tzset (void) attribute_hidden;

/* Prototype for the internal function to get information based on TZ.  */
extern struct tm *__tz_convert (__time64_t timer, int use_localtime,
				struct tm *tp) attribute_hidden;

extern int __nanosleep (const struct timespec *__requested_time,
			struct timespec *__remaining);
hidden_proto (__nanosleep)
#if __TIMESIZE == 64
# define __nanosleep64 __nanosleep
#else
extern int __nanosleep64 (const struct __timespec64 *__requested_time,
			  struct __timespec64 *__remaining);
hidden_proto (__nanosleep64)
#endif


extern int __getdate_r (const char *__string, struct tm *__resbufp);
libc_hidden_proto (__getdate_r);


/* Determine CLK_TCK value.  */
extern int __getclktck (void) attribute_hidden;


/* strptime support.  */
extern char * __strptime_internal (const char *rp, const char *fmt,
				   struct tm *tm, void *statep,
				   locale_t locparam) attribute_hidden;

#if __TIMESIZE == 64
# define __difftime64 __difftime
#else
extern double __difftime64 (__time64_t time1, __time64_t time0);
libc_hidden_proto (__difftime64)
#endif

extern double __difftime (time_t time1, time_t time0);

#if __TIMESIZE == 64
# define __clock_nanosleep_time64 __clock_nanosleep
# define __clock_gettime64 __clock_gettime
# define __timespec_get64 __timespec_get
# define __timespec_getres64 __timespec_getres
#else
extern int __clock_nanosleep_time64 (clockid_t clock_id,
                                     int flags, const struct __timespec64 *req,
                                     struct __timespec64 *rem);
libc_hidden_proto (__clock_nanosleep_time64)
extern int __clock_gettime64 (clockid_t clock_id, struct __timespec64 *tp);
libc_hidden_proto (__clock_gettime64)
extern int __timespec_get64 (struct __timespec64 *ts, int base);
libc_hidden_proto (__timespec_get64)
extern int __timespec_getres64 (struct __timespec64 *ts, int base);
libc_hidden_proto (__timespec_getres64)
#endif

#if __TIMESIZE == 64
# define __time64 __time
#else
extern __time64_t __time64 (__time64_t *timer);
libc_hidden_proto (__time64)
#endif

/* Use in the clock_* functions.  Size of the field representing the
   actual clock ID.  */
#define CLOCK_IDFIELD_SIZE	3

/* Check whether T fits in time_t.  */
static inline bool
in_time_t_range (__time64_t t)
{
  time_t s = t;
  return s == t;
}

/* Convert a known valid struct timeval into a struct __timespec64.  */
static inline struct __timespec64
valid_timeval_to_timespec64 (const struct timeval tv)
{
  struct __timespec64 ts64;

  ts64.tv_sec = tv.tv_sec;
  ts64.tv_nsec = tv.tv_usec * 1000;

  return ts64;
}

/* Convert a known valid struct timeval into a struct __timeval64.  */
static inline struct __timeval64
valid_timeval_to_timeval64 (const struct timeval tv)
{
  struct __timeval64 tv64;

  tv64.tv_sec = tv.tv_sec;
  tv64.tv_usec = tv.tv_usec;

  return tv64;
}

/* Convert a valid and within range of struct timeval, struct
   __timeval64 into a struct timeval.  */
static inline struct timeval
valid_timeval64_to_timeval (const struct __timeval64 tv64)
{
  struct timeval tv;

  tv.tv_sec = (time_t) tv64.tv_sec;
  tv.tv_usec = (suseconds_t) tv64.tv_usec;

  return tv;
}

/* Convert a struct __timeval64 into a struct __timespec64.  */
static inline struct __timespec64
timeval64_to_timespec64 (const struct __timeval64 tv64)
{
  struct __timespec64 ts64;

  ts64.tv_sec = tv64.tv_sec;
  ts64.tv_nsec = tv64.tv_usec * 1000;

  return ts64;
}

/* Convert a known valid struct timespec into a struct __timespec64.  */
static inline struct __timespec64
valid_timespec_to_timespec64 (const struct timespec ts)
{
  struct __timespec64 ts64;

  ts64.tv_sec = ts.tv_sec;
  ts64.tv_nsec = ts.tv_nsec;

  return ts64;
}

/* Convert a valid and within range of struct timespec, struct
   __timespec64 into a struct timespec.  */
static inline struct timespec
valid_timespec64_to_timespec (const struct __timespec64 ts64)
{
  struct timespec ts;

  ts.tv_sec = (time_t) ts64.tv_sec;
  ts.tv_nsec = ts64.tv_nsec;

  return ts;
}

/* Convert a valid and within range of struct timeval struct
   __timespec64 into a struct timeval.  */
static inline struct timeval
valid_timespec64_to_timeval (const struct __timespec64 ts64)
{
  struct timeval tv;

  tv.tv_sec = (time_t) ts64.tv_sec;
  tv.tv_usec = ts64.tv_nsec / 1000;

  return tv;
}

/* Convert a struct __timespec64 into a struct __timeval64.  */
static inline struct __timeval64
timespec64_to_timeval64 (const struct __timespec64 ts64)
{
  struct __timeval64 tv64;

  tv64.tv_sec = ts64.tv_sec;
  tv64.tv_usec = ts64.tv_nsec / 1000;

  return tv64;
}

/* A version of 'struct timeval' with 32-bit time_t
   and suseconds_t.  */
struct __timeval32
{
  __int32_t tv_sec;         /* Seconds.  */
  __int32_t tv_usec;        /* Microseconds.  */
};

/* Conversion functions for converting to/from __timeval32  */
static inline struct __timeval64
valid_timeval32_to_timeval64 (const struct __timeval32 tv)
{
  return (struct __timeval64) { tv.tv_sec, tv.tv_usec };
}

static inline struct __timeval32
valid_timeval64_to_timeval32 (const struct __timeval64 tv64)
{
  return (struct __timeval32) { tv64.tv_sec, tv64.tv_usec };
}

static inline struct timeval
valid_timeval32_to_timeval (const struct __timeval32 tv)
{
  return (struct timeval) { tv.tv_sec, tv.tv_usec };
}

static inline struct __timeval32
valid_timeval_to_timeval32 (const struct timeval tv)
{
  return (struct __timeval32) { tv.tv_sec, tv.tv_usec };
}

static inline struct timespec
valid_timeval32_to_timespec (const struct __timeval32 tv)
{
  return (struct timespec) { tv.tv_sec, tv.tv_usec * 1000 };
}

static inline struct __timeval32
valid_timespec_to_timeval32 (const struct timespec ts)
{
  return (struct __timeval32) { (time_t) ts.tv_sec, ts.tv_nsec / 1000 };
}

static inline struct __timeval64
valid_timespec_to_timeval64 (const struct timespec ts)
{
  return (struct __timeval64) { (time_t) ts.tv_sec, ts.tv_nsec / 1000 };
}

/* Check if a value is in the valid nanoseconds range. Return true if
   it is, false otherwise.  */
static inline bool
valid_nanoseconds (__syscall_slong_t ns)
{
  return __glibc_likely (0 <= ns && ns < 1000000000);
}

/* Helper function to get time in seconds, similar to time.  */
static inline time_t
time_now (void)
{
  struct timespec ts;
  __clock_gettime (TIME_CLOCK_GETTIME_CLOCKID, &ts);
  return ts.tv_sec;
}

static inline __time64_t
time64_now (void)
{
  struct __timespec64 ts;
  __clock_gettime64 (TIME_CLOCK_GETTIME_CLOCKID, &ts);
  return ts.tv_sec;
}

#define NSEC_PER_SEC    1000000000L  /* Nanoseconds per second.  */
#define USEC_PER_SEC    1000000L     /* Microseconds per second.  */
#define NSEC_PER_USEC   1000L        /* Nanoseconds per microsecond.  */

#endif

#endif
