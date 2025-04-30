/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#include <ctype.h>
#include <fnmatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>

#define HEADER_MAX          256

static char macrofile[] = "/tmp/annexc.XXXXXX";

/* <aio.h>.  */
static const char *const aio_syms[] =
{
  "AIO_ALLDONE", "AIO_CANCELED", "AIO_NOTCANCELED", "LIO_NOP", "LIO_NOWAIT",
  "LIO_READ", "LIO_WAIT", "LIO_WRITE",
  /* From <fcntl.h>.  */
  "FD_CLOEXEC", "F_DUPFD", "F_GETFD", "F_GETFL", "F_GETLK", "F_RDLCK",
  "F_SETFD", "F_SETFL", "F_SETLK", "F_SETLKW", "F_UNLCK", "F_WRLCK",
  "O_ACCMODE", "O_APPEND", "O_CREAT", "O_DSYNC", "O_EXCL", "O_NOCTTY",
  "O_NONBLOCK", "O_RDONLY", "O_RDWR", "O_RSYNC", "O_SYNC", "O_TRUNC",
  "O_WRONLY",
  /* From <signal.h>.  */
  "SA_NOCLDSTOP", "SA_SIGINFO", "SIGABRT", "SIGALRM", "SIGBUS", "SIGCHLD",
  "SIGCONT", "SIGEV_NONE", "SIGEV_SIGNAL", "SIGEV_SIGNAL", "SIGEV_THREAD",
  "SIGFPE", "SIGHUP", "SIGILL", "SIGINT", "SIGKILL", "SIGPIPE", "SIGQUIT",
  "SIGRTMAX", "SIGRTMIN", "SIGSEGV", "SIGSTOP", "SIGTERM", "SIGTSTP",
  "SIGTTIN", "SIGTTOU", "SIGUSR1", "SIGUSR2", "SIG_BLOCK", "SIG_DFL",
  "SIG_ERR", "SIG_IGN", "SIG_SETMASK", "SIG_UNBLOCK", "SI_ASYNCIO",
  "SI_MESGQ", "SI_QUEUE", "SI_TIMER", "SI_USER"
};
static const char *const aio_maybe[] =
{
  "aio_cancel", "aio_error", "aio_fsync", "aio_read", "aio_return",
  "aio_suspend", "aio_write", "lio_listio",
  /* From <fcntl.h>.  */
  "creat", "fcntl", "open", "SEEK_CUR", "SEEK_END", "SEEK_SET", "S_IRGRP",
  "S_IROTH", "S_IRUSR", "S_IRWXG", "S_IRWXO", "S_IRWXU", "S_ISBLK",
  "S_ISCHR", "S_ISDIR", "S_ISFIFO", "S_ISGID", "S_ISREG", "S_ISUID",
  "S_IWGRP", "S_IWOTH", "S_IWUSR", "S_IXGRP", "S_IXOTH", "S_IXUSR",
  /* From <signal.h>.  */
  "kill", "raise", "sigaction", "sigaddset", "sigdelset", "sigemptyset",
  "sigfillset", "sigismember", "signal", "sigpending", "sigprocmask",
  "sigqueue", "sigsuspend", "sigtimedwait", "sigwait", "sigwaitinfo"
};

/* <assert.h>.  */
static const char *const assert_syms[] =
{
  "assert"
};
static const char *const assert_maybe[] =
{
};

/* <ctype.h>.   */
static const char *const ctype_syms[] =
{
};
static const char *const ctype_maybe[] =
{
  "isalnum", "isalpha", "iscntrl", "isdigit", "isgraph", "islower",
  "isprint", "ispunct", "isspace", "isupper", "isxdigit", "tolower",
  "toupper"
};

/* <dirent.h>.  */
static const char *const dirent_syms[] =
{
};
static const char *const dirent_maybe[] =
{
  "closedir", "opendir", "readdir", "readdir_r", "rewinddir"
};

/* <errno.h>.  */
static const char *const errno_syms[] =
{
  "E2BIG", "EACCES", "EAGAIN", "EBADF", "EBADMSG", "EBUSY", "ECANCELED",
  "ECHILD", "EDEADLK", "EDOM", "EEXIST", "EFAULT", "EFBIG", "EINPROGRESS",
  "EINTR", "EINVAL", "EIO", "EISDIR", "EMFILE", "EMLINK", "EMSGSIZE",
  "ENAMETOOLONG", "ENFILE", "ENODEV", "ENOENT", "ENOEXEC", "ENOLCK",
  "ENOMEM", "ENOSPC", "ENOSYS", "ENOTDIR", "ENOTEMPTY", "ENOTSUP",
  "ENOTTY", "ENXIO", "EPERM", "EPIPE", "ERANGE", "EROFS", "ESPIPE",
  "ESRCH", "ETIMEDOUT", "EXDEV"
};
static const char *const errno_maybe[] =
{
  "errno", "E*"
};

/* <fcntl.h>.  */
static const char *const fcntl_syms[] =
{
  "FD_CLOEXEC", "F_DUPFD", "F_GETFD", "F_GETFL", "F_GETLK", "F_RDLCK",
  "F_SETFD", "F_SETFL", "F_SETLK", "F_SETLKW", "F_UNLCK", "F_WRLCK",
  "O_ACCMODE", "O_APPEND", "O_CREAT", "O_DSYNC", "O_EXCL", "O_NOCTTY",
  "O_NONBLOCK", "O_RDONLY", "O_RDWR", "O_RSYNC", "O_SYNC", "O_TRUNC",
  "O_WRONLY"
};
static const char *const fcntl_maybe[] =
{
  "creat", "fcntl", "open", "SEEK_CUR", "SEEK_END", "SEEK_SET", "S_IRGRP",
  "S_IROTH", "S_IRUSR", "S_IRWXG", "S_IRWXO", "S_IRWXU", "S_ISBLK",
  "S_ISCHR", "S_ISDIR", "S_ISFIFO", "S_ISGID", "S_ISREG", "S_ISUID",
  "S_IWGRP", "S_IWOTH", "S_IWUSR", "S_IXGRP", "S_IXOTH", "S_IXUSR"
};

/* <float.h>.  */
static const char *const float_syms[] =
{
  "DBL_DIG", "DBL_EPSILON", "DBL_MANT_DIG", "DBL_MAX", "DBL_MAX_10_EXP",
  "DBL_MAX_EXP", "DBL_MIN", "DBL_MIN_10_EXP", "DBL_MIN_EXP", "FLT_DIG",
  "FLT_EPSILON", "FLT_MANT_DIG", "FLT_MAX", "FLT_MAX_10_EXP", "FLT_MAX_EXP",
  "FLT_MIN", "FLT_MIN_10_EXP", "FLT_MIN_EXP", "FLT_RADIX", "FLT_ROUNDS",
  "LDBL_DIG", "LDBL_EPSILON", "LDBL_MANT_DIG", "LDBL_MAX", "LDBL_MAX_10_EXP",
  "LDBL_MAX_EXP", "LDBL_MIN", "LDBL_MIN_10_EXP", "LDBL_MIN_EXP"
};
static const char *const float_maybe[] =
{
};

/* <grp.h>.  */
static const char *const grp_syms[] =
{
};
static const char *const grp_maybe[] =
{
  "getgrgid", "getgrgid_r", "getgrnam", "getgrnam_r"
};

/* <limits.h>.  */
static const char *const limits_syms[] =
{
  "_POSIX_AIO_LISTIO_MAX", "_POSIX_AIO_MAX", "_POSIX_ARG_MAX",
  "_POSIX_CHILD_MAX", "_POSIX_CLOCKRES_MAX", "_POSIX_DELAYTIMER_MAX",
  "_POSIX_LINK_MAX", "_POSIX_LOGIN_NAME_MAX", "_POSIX_MAX_CANON",
  "_POSIX_MAX_INPUT", "_POSIX_MQ_OPEN_MAX", "_POSIX_MQ_PRIO_MAX",
  "_POSIX_NAME_MAX", "_POSIX_NGROUPS_MAX", "_POSIX_OPEN_MAX",
  "_POSIX_PATH_MAX", "_POSIX_PIPE_BUF", "_POSIX_RTSIG_MAX",
  "_POSIX_SEM_NSEMS_MAX", "_POSIX_SEM_VALUE_MAX", "_POSIX_SIGQUEUE_MAX",
  "_POSIX_SSIZE_MAX", "_POSIX_STREAM_MAX",
  "_POSIX_THREAD_DESTRUCTOR_ITERATIONS", "_POSIX_THREAD_KEYS_MAX",
  "_POSIX_THREAD_THREADS_MAX", "_POSIX_TIMER_MAX", "_POSIX_TTY_NAME_MAX",
  "_POSIX_TZNAME_MAX", "_POSIX_THREAD_DESTRUCTOR_ITERATIONS",
  "CHAR_BIT", "CHAR_MAX", "CHAR_MIN", "INT_MAX", "INT_MIN", "LONG_MAX",
  "LONG_MIN", "MB_LEN_MAX", "NGROUPS_MAX", "PAGESIZE", "SCHAR_MAX",
  "SCHAR_MIN", "SHRT_MAX", "SHRT_MIN", "UCHAR_MAX", "UINT_MAX",
  "ULONG_MAX", "USHRT_MAX"
};
static const char *const limits_maybe[] =
{
  "AIO_LISTIO_MAX", "AIO_MAX", "ARG_MAX", "CHILD_MAX", "DELAYTIMER_MAX",
  "LINK_MAX", "LOGIN_NAME_MAX", "LONG_MAX", "LONG_MIN", "MAX_CANON",
  "MAX_INPUT", "MQ_OPEN_MAX", "MQ_PRIO_MAX", "NAME_MAX", "OPEN_MAX",
  "PATH_MAX", "PIPE_BUF", "RTSIG_MAX", "PTHREAD_DESTRUCTOR_ITERATIONS",
  "PTHREAD_KEYS_MAX", "PTHREAD_STACK_MIN", "PTHREAD_THREADS_MAX"
};

/* <locale.h>.  */
static const char *const locale_syms[] =
{
  "LC_ALL", "LC_COLLATE", "LC_CTYPE", "LC_MONETARY", "LC_NUMERIC",
  "LC_TIME", "NULL"
};
static const char *const locale_maybe[] =
{
  "LC_*", "localeconv", "setlocale"
};

/* <math.h>.  */
static const char *const math_syms[] =
{
  "HUGE_VAL"
};
static const char *const math_maybe[] =
{
  "acos", "asin", "atan2", "atan", "ceil", "cos", "cosh", "exp",
  "fabs", "floor", "fmod", "frexp", "ldexp", "log10", "log", "modf",
  "pow", "sin", "sinh", "sqrt", "tan", "tanh",
  "acosf", "asinf", "atan2f", "atanf", "ceilf", "cosf", "coshf", "expf",
  "fabsf", "floorf", "fmodf", "frexpf", "ldexpf", "log10f", "logf", "modff",
  "powf", "sinf", "sinhf", "sqrtf", "tanf", "tanhf",
  "acosl", "asinl", "atan2l", "atanl", "ceill", "cosl", "coshl", "expl",
  "fabsl", "floorl", "fmodl", "frexpl", "ldexpl", "log10l", "logl", "modfl",
  "powl", "sinl", "sinhl", "sqrtl", "tanl", "tanhl"
};

/* <mqueue.h>.  */
static const char *const mqueue_syms[] =
{
};
static const char *const mqueue_maybe[] =
{
  "mq_close", "mq_getattr", "mq_notify", "mq_open", "mq_receive",
  "mq_send", "mq_setattr", "mq_unlink"
};

/* <pthread.h>.  */
static const char *const pthread_syms[] =
{
  "PTHREAD_CANCELED", "PTHREAD_CANCEL_ASYNCHRONOUS",
  "PTHREAD_CANCEL_DEFERRED", "PTHREAD_CANCEL_DISABLE", "PTHREAD_CANCEL_ENABLE",
  "PTHREAD_COND_INITIALIZER", "PTHREAD_CREATE_DETACHED",
  "PTHREAD_CREATE_JOINABLE", "PTHREAD_EXPLICIT_SCHED",
  "PTHREAD_INHERIT_SCHED", "PTHREAD_MUTEX_INITIALIZER",
  "PTHREAD_ONCE_INIT", "PTHREAD_PRIO_INHERIT", "PTHREAD_PRIO_NONE",
  "PTHREAD_PRIO_PROTECT", "PTHREAD_PROCESS_PRIVATE",
  "PTHREAD_PROCESS_SHARED", "PTHREAD_SCOPE_PROCESS", "PTHREAD_SCOPE_SYSTEM",
  /* These come from <sched.h>.  */
  "SCHED_FIFO", "SCHED_OTHER", "SCHED_RR",
  /* These come from <time.h>.  */
  "CLK_TCK", "CLOCKS_PER_SEC", "CLOCK_REALTIME", "NULL", "TIMER_ABSTIME"
};
static const char *const pthread_maybe[] =
{
  "pthread_atfork", "pthread_attr_destroy", "pthread_attr_getdetachstate",
  "pthread_attr_getinheritsched", "pthread_attr_getschedparam",
  "pthread_attr_getschedpolicy", "pthread_attr_getscope",
  "pthread_attr_getstackaddr", "pthread_attr_getstacksize",
  "pthread_attr_init", "pthread_attr_setdetachstate",
  "pthread_attr_setinheritsched", "pthread_attr_setschedparam",
  "pthread_attr_setschedpolicy", "pthread_attr_setscope",
  "pthread_attr_setstackaddr", "pthread_attr_setstacksize",
  "pthread_cleanup_pop", "pthread_cleanup_push", "pthread_cond_broadcast",
  "pthread_cond_destroy", "pthread_cond_init", "pthread_cond_signal",
  "pthread_cond_timedwait", "pthread_cond_wait", "pthread_condattr_destroy",
  "pthread_condattr_getpshared", "pthread_condattr_init",
  "pthread_condattr_setpshared", "pthread_create", "pthread_detach",
  "pthread_equal", "pthread_exit", "pthread_getspecific", "pthread_join",
  "pthread_key_create", "pthread_key_destroy", "pthread_kill",
  "pthread_mutex_destroy", "pthread_mutex_getprioceiling",
  "pthread_mutex_init", "pthread_mutex_lock", "pthread_mutex_setprioceiling",
  "pthread_mutex_trylock", "pthread_mutex_unlock", "pthread_mutexattr_destroy",
  "pthread_mutexattr_getprioceiling", "pthread_mutexattr_getprotocol",
  "pthread_mutexattr_getpshared", "pthread_mutexattr_init",
  "pthread_mutexattr_setprioceiling", "pthread_mutexattr_setprotocol",
  "pthread_mutexattr_setpshared", "pthread_once", "pthread_self",
  "pthread_setcancelstate", "pthread_setcanceltype", "pthread_setspecific",
  "pthread_sigmask", "pthread_testcancel"
  /* These come from <sched.h>.  */
  "sched_get_priority_max", "sched_get_priority_min",
  "sched_get_rr_interval", "sched_getparam", "sched_getscheduler",
  "sched_setparam", "sched_setscheduler", "sched_yield",
  /* These come from <time.h>.  */
  "asctime", "asctime_r", "clock", "clock_getres", "clock_gettime",
  "clock_settime", "ctime", "ctime_r", "difftime", "gmtime", "gmtime_r",
  "localtime", "localtime_r", "mktime", "nanosleep", "strftime", "time",
  "timer_create", "timer_delete", "timer_getoverrun", "timer_gettime",
  "timer_settime", "tzset"
};

/* <pwd.h>.  */
static const char *const pwd_syms[] =
{
};
static const char *const pwd_maybe[] =
{
  "getpwnam", "getpwnam_r", "getpwuid", "getpwuid_r"
};

/* <sched.h>.  */
static const char *const sched_syms[] =
{
  "SCHED_FIFO", "SCHED_OTHER", "SCHED_RR",
};
static const char *const sched_maybe[] =
{
  "sched_get_priority_max", "sched_get_priority_min",
  "sched_get_rr_interval", "sched_getparam", "sched_getscheduler",
  "sched_setparam", "sched_setscheduler", "sched_yield",
  /* These come from <time.h>.  */
  "CLK_TCK", "CLOCKS_PER_SEC", "CLOCK_REALTIME", "NULL", "TIMER_ABSTIME"
  "asctime", "asctime_r", "clock", "clock_getres", "clock_gettime",
  "clock_settime", "ctime", "ctime_r", "difftime", "gmtime", "gmtime_r",
  "localtime", "localtime_r", "mktime", "nanosleep", "strftime", "time",
  "timer_create", "timer_delete", "timer_getoverrun", "timer_gettime",
  "timer_settime", "tzset"
};

/* <semaphore.h>.  */
static const char *const semaphore_syms[] =
{
};
static const char *const semaphore_maybe[] =
{
  "sem_close", "sem_destroy", "sem_getvalue", "sem_init", "sem_open",
  "sen_post", "sem_trywait", "sem_unlink", "sem_wait"
};

/* <setjmp.h>.  */
static const char *const setjmp_syms[] =
{
};
static const char *const setjmp_maybe[] =
{
  "longjmp", "setjmp", "siglongjmp", "sigsetjmp"
};

/* <signal.h>.  */
static const char *const signal_syms[] =
{
  "SA_NOCLDSTOP", "SA_SIGINFO", "SIGABRT", "SIGALRM", "SIGBUS", "SIGCHLD",
  "SIGCONT", "SIGEV_NONE", "SIGEV_SIGNAL", "SIGEV_THREAD",
  "SIGFPE", "SIGHUP", "SIGILL", "SIGINT", "SIGKILL", "SIGPIPE", "SIGQUIT",
  "SIGRTMAX", "SIGRTMIN", "SIGSEGV", "SIGSTOP", "SIGTERM", "SIGTSTP",
  "SIGTTIN", "SIGTTOU", "SIGUSR1", "SIGUSR2", "SIG_BLOCK", "SIG_DFL",
  "SIG_ERR", "SIG_IGN", "SIG_SETMASK", "SIG_UNBLOCK", "SI_ASYNCIO",
  "SI_MESGQ", "SI_QUEUE", "SI_TIMER", "SI_USER"
};
static const char *const signal_maybe[] =
{
  "kill", "raise", "sigaction", "sigaddset", "sigdelset", "sigemptyset",
  "sigfillset", "sigismember", "signal", "sigpending", "sigprocmask",
  "sigqueue", "sigsuspend", "sigtimedwait", "sigwait", "sigwaitinfo"
};

/* <stdarg.h>.  */
static const char *const stdarg_syms[] =
{
  "va_arg", "va_end", "va_start"
};
static const char *const stdarg_maybe[] =
{
  "va_list"
};

/* <stddef.h>.  */
static const char *const stddef_syms[] =
{
  "NULL", "offsetof"
};
static const char *const stddef_maybe[] =
{
};

/* <stdio.h>.  */
static const char *const stdio_syms[] =
{
  "BUFSIZ", "EOF", "FILENAME_MAX", "FOPEN_MAX", "L_ctermid", "L_cuserid",
  "L_tmpnam", "NULL", "SEEK_CUR", "SEEK_END", "SEEK_SET", "STREAM_MAX",
  "TMP_MAX", "stderr", "stdin", "stdout", "_IOFBF", "_IOLBF", "_IONBF"
};
static const char *const stdio_maybe[] =
{
  "clearerr", "fclose", "fdopen", "feof", "ferror", "fflush", "fgetc",
  "fgetpos", "fgets", "fileno", "flockfile", "fopen", "fprintf", "fputc",
  "fputs", "fread", "freopen", "fscanf", "fseek", "fsetpos", "ftell",
  "ftrylockfile", "funlockfile", "fwrite", "getc", "getchar",
  "getchar_unlocked", "getc_unlocked", "gets", "perror", "printf", "putc",
  "putchar", "putchar_unlocked", "putc_unlocked", "puts", "remove", "rename",
  "rewind", "scanf", "setbuf", "setvbuf", "sprintf", "sscanf", "tmpfile",
  "tmpnam", "ungetc", "vfprintf", "vprintf", "vsprintf"
};

/* <stdlib.h>.  */
static const char *const stdlib_syms[] =
{
  "EXIT_FAILURE", "EXIT_SUCCESS", "MB_CUR_MAX", "NULL", "RAND_MAX"
};
static const char *const stdlib_maybe[] =
{
  "abort", "abs", "atexit", "atof", "atoi", "atol", "bsearch", "calloc",
  "div", "exit", "free", "getenv", "labs", "ldiv", "malloc", "mblen",
  "mbstowcs", "mbtowc", "qsort", "rand", "rand_r", "realloc", "srand",
  "strtod", "strtol", "strtoul", "system", "wcstombs", "wctomb"
};

/* <string.h>.  */
static const char *const string_syms[] =
{
  "NULL"
};
static const char *const string_maybe[] =
{
  "memchr", "memcmp", "memcpy", "memmove", "memset", "strcat", "strchr",
  "strcmp", "strcoll", "strcpy", "strcspn", "strerror", "strlen",
  "strncat", "strncmp", "strncpy", "strpbrk", "strrchr", "strspn",
  "strstr", "strtok", "strtok_r", "strxfrm"
};

/* <sys/mman.h>.  */
static const char *const mman_syms[] =
{
  "MAP_FAILED", "MAP_FIXED", "MAP_PRIVATE", "MAP_SHARED", "MCL_CURRENT",
  "MCL_FUTURE", "MS_ASYNC", "MS_INVALIDATE", "MS_SYNC", "PROT_EXEC",
  "PROT_NONE", "PROT_READ", "PROT_WRITE"
};
static const char *const mman_maybe[] =
{
  "mlock", "mlockall", "mmap", "mprotect", "msync", "munlock", "munlockall",
  "munmap", "shm_open", "shm_unlock"
};

/* <sys/stat.h>.  */
static const char *const stat_syms[] =
{
  "S_IRGRP", "S_IROTH", "S_IRUSR", "S_IRWXG", "S_IRWXO", "S_IRWXU",
  "S_ISBLK", "S_ISCHR", "S_ISDIR", "S_ISFIFO", "S_ISGID", "S_ISREG",
  "S_ISUID", "S_IWGRP", "S_IWOTH", "S_IWUSR", "S_IXGRP", "S_IXOTH",
  "S_IXUSR", "S_TYPEISMQ", "S_TYPEISSEM", "S_TYPEISSHM"
};
static const char *const stat_maybe[] =
{
  "chmod", "fchmod", "fstat", "mkdir", "mkfifo", "stat", "umask"
};

/* <sys/times.h>.  */
static const char *const times_syms[] =
{
};
static const char *const times_maybe[] =
{
  "times"
};

/* <sys/types.h>.  */
static const char *const types_syms[] =
{
};
static const char *const types_maybe[] =
{
};

/* <sys/utsname.h>.  */
static const char *const utsname_syms[] =
{
};
static const char *const utsname_maybe[] =
{
  "uname"
};

/* <sys/wait.h>.  */
static const char *const wait_syms[] =
{
  "WEXITSTATUS", "WIFEXITED", "WIFSIGNALED", "WIFSTOPPED", "WNOHANG",
  "WSTOPSIG", "WTERMSIG", "WUNTRACED"
};
static const char *const wait_maybe[] =
{
  "wait", "waitpid"
};

/* <termios.h>.  */
static const char *const termios_syms[] =
{
  "B0", "B110", "B1200", "B134", "B150", "B1800", "B19200", "B200", "B2400",
  "B300", "B38400", "B4800", "B50", "B600", "B75", "B9600", "BRKINT", "CLOCAL",
  "CREAD", "CS5", "CS6", "CS7", "CS8", "CSIZE", "CSTOPN", "ECHO", "ECHOE",
  "ECHOK", "ECHONL", "HUPCL", "ICANON", "ICRNL", "IEXTEN", "IGNBRK", "IGNCR",
  "IGNPAR", "INCLR", "INPCK", "ISIG", "ISTRIP", "IXOFF", "IXON", "NCCS",
  "NOFLSH", "OPOST", "PARENB", "PARMRK", "PARODD", "TCIFLUSH", "TCIOFF",
  "TCIOFLUSH", "TCOFLUSH", "TCOOFF", "TCOON", "TCSADRAIN", "TCSAFLUSH",
  "TCSANOW", "TOSTOP", "VEOF", "VEOL", "VERASE", "VINTR", "VKILL", "VMIN",
  "VQUIT", "VSTART", "VSTOP", "VSUSP", "VTIME"
};
static const char *const termios_maybe[] =
{
  "cfgetispeed", "cfgetospeed", "cfsetispeed", "cfsetospeed", "tcdrain",
  "tcflow", "tcflush", "tcgetattr", "tcsendbrk", "tcsetattr"
};

/* <time.h>.  */
static const char *const time_syms[] =
{
  "CLK_TCK", "CLOCKS_PER_SEC", "CLOCK_REALTIME", "NULL", "TIMER_ABSTIME"
};
static const char *const time_maybe[] =
{
  "asctime", "asctime_r", "clock", "clock_getres", "clock_gettime",
  "clock_settime", "ctime", "ctime_r", "difftime", "gmtime", "gmtime_r",
  "localtime", "localtime_r", "mktime", "nanosleep", "strftime", "time",
  "timer_create", "timer_delete", "timer_getoverrun", "timer_gettime",
  "timer_settime", "tzset"
};

/* <unistd.h>.  */
static const char *const unistd_syms[] =
{
  "F_OK", "NULL", "R_OK", "SEEK_CUR", "SEEK_END", "SEEK_SET", "STDERR_FILENO",
  "STDIN_FILENO", "STDOUT_FILENO", "W_OK", "X_OK",
  "_PC_ASYNC_IO", "_PC_CHOWN_RESTRICTED", "_PC_LINK_MAX", "_PC_MAX_CANON",
  "_PC_MAX_INPUT", "_PC_NAME_MAX", "_PC_NO_TRUNC", "_PC_PATH_MAX",
  "_PC_PIPE_BUF", "_PC_PRIO_IO", "_PC_SYNC_IO", "_PC_VDISABLE",
  "_SC_AIO_LISTIO_MAX", "_SC_AIO_MAX", "_SC_AIO_PRIO_DELTA_MAX",
  "_SC_ARG_MAX", "_SC_ASYNCHRONOUS_IO", "_SC_CHILD_MAX", "_SC_CLK_TCK",
  "_SC_DELAYTIMER_MAX", "_SC_FSYNC", "_SC_GETGR_R_SIZE_MAX",
  "_SC_GETPW_R_SIZE_MAX", "_SC_JOB_CONTROL", "_SC_LOGIN_NAME_MAX",
  "_SC_MAPPED_FILES", "_SC_MEMLOCK", "_SC_MEMLOCK_RANGE",
  "_SC_MEMORY_PROTECTION", "_SC_MESSAGE_PASSING", "_SC_MQ_OPEN_MAX",
  "_SC_MQ_PRIO_MAX", "_SC_NGROUPS_MAX", "_SC_OPEN_MAX", "_SC_PAGESIZE",
  "_SC_PRIORITIZED_IO", "_SC_PRIORITY_SCHEDULING", "_SC_REALTIME_SIGNALS",
  "_SC_RTSIG_MAX", "_SC_SAVED_IDS", "_SC_SEMAPHORES", "_SC_SEM_NSEMS_MAX",
  "_SC_SEM_VALUE_MAX", "_SC_SHARED_MEMORY_OBJECTS", "_SC_SIGQUEUE_MAX",
  "_SC_STREAM_MAX", "_SC_SYNCHRONIZED_IO", "_SC_THREADS",
  "_SC_THREAD_ATTR_STACKADDR", "_SC_THREAD_ATTR_STACKSIZE",
  "_SC_THREAD_DESTRUCTOR_ITERATIONS", "_SC_THREAD_PRIO_INHERIT",
  "_SC_THREAD_PRIORITY_SCHEDULING", "_SC_THREAD_PRIO_PROTECT",
  "_SC_THREAD_PROCESS_SHARED", "_SC_THREAD_SAFE_FUNCTIONS",
  "_SC_THREAD_STACK_MIN", "_SC_THREAD_THREADS_MAX", "_SC_TIMERS",
  "_SC_TIMER_MAX", "_SC_TTY_NAME_MAX", "_SC_TZNAME_MAX", "_SC_VERSION"
};
static const char *const unistd_maybe[] =
{
  "_POSIX_ASYNCHRONOUS_IO", "_POSIX_ASYNC_IO", "_POSIX_CHOWN_RESTRICTED",
  "_POSIX_FSYNC", "_POSIX_JOB_CONTROL", "_POSIX_MAPPED_FILES",
  "_POSIX_MEMLOCK", "_POSIX_MEMLOCK_RANGE", "_MEMORY_PROTECTION",
  "_POSIX_MESSAGE_PASSING", "_POSIX_NO_TRUNC", "_POSIX_PRIORITIZED_IO",
  "_POSIX_PRIORITY_SCHEDULING", "_POSIX_PRIO_IO", "_POSIX_REATIME_SIGNALS",
  "_POSIX_SAVED_IDS", "_POSIX_SEMAPHORES", "_POSIX_SHARED_MEMORY_OBJECTS",
  "_POSIX_SYNCHRONIZED_IO", "_POSIX_SYNC_IO", "_POSIX_THREADS",
  "_POSIX_THREAD_ATTR_STACKADDR", "_POSIX_THREAD_ATTR_STACKSIZE",
  "_POSIX_THREAD_PRIO_INHERIT", "_POSIX_THREAD_PRIO_PROTECT",
  "_POSIX_THREAD_PROCESS_SHARED", "_POSIX_THREAD_SAFE_FUNCTIONS",
  "_POSIX_THREAD_PRIORITY_SCHEDULING", "_POSIX_TIMERS",
  "_POSIX_VDISABLE", "_POSIX_VERSION",
  "_exit", "access", "alarm", "chdir", "chown", "close", "ctermid", "cuserid",
  "dup2", "dup", "execl", "execle", "execlp", "execv", "execve", "execvp",
  "fdatasync", "fork", "fpathconf", "fsync", "ftruncate", "getcwd", "getegid",
  "geteuid", "getgid", "getgroups", "getlogin", "getlogin_r", "getpgrp",
  "getpid", "getppid", "getuid", "isatty", "link", "lseek", "pathconf",
  "pause", "pipe", "read", "rmdir", "setgid", "setgpid", "setsid", "setuid",
  "sleep", "sleep", "sysconf", "tcgetpgrp", "tcsetpgrp", "ttyname",
  "ttyname_r", "unlink", "write"
};

/* <utime.h>.  */
static const char *const utime_syms[] =
{
};
static const char *const utime_maybe[] =
{
  "utime"
};


static struct header
{
  const char *name;
  const char *const *syms;
  size_t nsyms;
  const char *const *maybe;
  size_t nmaybe;
  const char *subset;
} headers[] =
{
#define H(n) \
  { #n ".h", n##_syms, sizeof (n##_syms) / sizeof (n##_syms[0]), \
    n##_maybe, sizeof (n##_maybe) / sizeof (n##_maybe[0]), NULL }
#define Hc(n, s) \
  { #n ".h", n##_syms, sizeof (n##_syms) / sizeof (n##_syms[0]), \
    n##_maybe, sizeof (n##_maybe) / sizeof (n##_maybe[0]), s }
#define Hs(n) \
  { "sys/" #n ".h", n##_syms, sizeof (n##_syms) / sizeof (n##_syms[0]), \
    n##_maybe, sizeof (n##_maybe) / sizeof (n##_maybe[0]), NULL }
  H(aio),
  H(assert),
  H(ctype),
  H(dirent),
  H(errno),
  H(fcntl),
  H(float),
  H(grp),
  H(limits),
  H(locale),
  H(math),
  Hc(mqueue, "_POSIX_MESSAGE_PASSING"),
  H(pthread),
  H(pwd),
  H(sched),
  H(semaphore),
  H(setjmp),
  H(signal),
  H(stdarg),
  H(stddef),
  H(stdio),
  H(stdlib),
  H(string),
  Hs(mman),
  Hs(stat),
  Hs(times),
  Hs(types),
  Hs(utsname),
  Hs(wait),
  H(termios),
  H(time),
  H(unistd),
  H(utime)
};

#define NUMBER_OF_HEADERS              (sizeof headers / sizeof *headers)


/* Format string to build command to invoke compiler.  */
static const char fmt[] = "\
echo \"#include <%s>\" |\
%s -E -dM -D_POSIX_SOURCE %s \
-isystem `%s --print-prog-name=include` - > %s";

static const char testfmt[] = "\
echo \"#include <unistd.h>\n#if !defined %s || %s == -1\n#error not defined\n#endif\n\" |\
%s -E -dM -D_POSIX_SOURCE %s \
-isystem `%s --print-prog-name=include` - 2> /dev/null > %s";


/* The compiler we use (given on the command line).  */
const char *CC;
/* The -I parameters for CC to find all headers.  */
const char *INC;

static char *xstrndup (const char *, size_t);
static const char **get_null_defines (void);
static int check_header (const struct header *, const char **);
static int xsystem (const char *);

int
main (int argc, char *argv[])
{
  int h;
  int result = 0;
  const char **ignore_list;

  CC = argc > 1 ? argv[1] : "gcc";
  INC = argc > 2 ? argv[2] : "";

  if (system (NULL) == 0)
    {
      puts ("Sorry, no command processor.");
      return EXIT_FAILURE;
    }

  /* First get list of symbols which are defined by the compiler.  */
  ignore_list = get_null_defines ();

  fputs ("Tested files:\n", stdout);

  for (h = 0; h < NUMBER_OF_HEADERS; ++h)
    result |= check_header (&headers[h], ignore_list);

  remove (macrofile);

  /* The test suite should return errors but for now this is not
     practical.  Give a warning and ask the user to correct the bugs.  */
  return result;
}


static char *
xstrndup (const char *s, size_t n)
{
  size_t len = n;
  char *new = malloc (len + 1);

  if (new == NULL)
    return NULL;

  new[len] = '\0';
  return memcpy (new, s, len);
}


/* Like system but propagate interrupt and quit signals.  */
int
xsystem (const char *cmd)
{
  int status;

  status = system (cmd);
  if (status != -1)
    {
      if (WIFSIGNALED (status))
	{
	  if (WTERMSIG (status) == SIGINT || WTERMSIG (status) == SIGQUIT)
	    raise (WTERMSIG (status));
	}
      else if (WIFEXITED (status))
	{
	  if (WEXITSTATUS (status) == SIGINT + 128
	      || WEXITSTATUS (status) == SIGQUIT + 128)
	    raise (WEXITSTATUS (status) - 128);
	}
    }
  return status;
}


static const char **
get_null_defines (void)
{
  char line[BUFSIZ], *command;
  char **result = NULL;
  size_t result_len = 0;
  size_t result_max = 0;
  FILE *input;
  int first = 1;

  int fd = mkstemp (macrofile);
  if (fd == -1)
    {
      printf ("mkstemp failed: %m\n");
      exit (1);
    }
  close (fd);

  command = malloc (sizeof fmt + sizeof "/dev/null" + 2 * strlen (CC)
		    + strlen (INC) + strlen (macrofile));

  if (command == NULL)
    {
      puts ("No more memory.");
      exit (1);
    }

  sprintf (command, fmt, "/dev/null", CC, INC, CC, macrofile);

  if (xsystem (command))
    {
      puts ("system() returned nonzero");
      return NULL;
    }
  free (command);
  input = fopen (macrofile, "r");

  if (input == NULL)
    {
      printf ("Could not read %s: ", macrofile);
      perror (NULL);
      return NULL;
    }

  while (fgets (line, sizeof line, input) != NULL)
    {
      char *start;
      if (strlen (line) < 9 || line[7] != ' ')
	{ /* "#define A" */
	  printf ("Malformed input, expected '#define MACRO'\ngot '%s'\n",
		  line);
	  continue;
	}
      if (line[8] == '_')
	/* It's a safe identifier.  */
	continue;
      if (result_len == result_max)
	{
	  result_max += 10;
	  result = realloc (result, result_max * sizeof (char **));
	  if (result == NULL)
	    {
	      puts ("No more memory.");
	      exit (1);
	    }
	}
      start = &line[8];
      result[result_len++] = xstrndup (start, strcspn (start, " ("));

      if (first)
	{
	  fputs ("The following identifiers will be ignored since the compiler defines them\nby default:\n", stdout);
	  first = 0;
	}
      puts (result[result_len - 1]);
    }
  if (result_len == result_max)
    {
      result_max += 1;
      result = realloc (result, result_max * sizeof (char **));
      if (result == NULL)
	{
	  puts ("No more memory.");
	  exit (1);
	}
    }
  result[result_len] = NULL;
  fclose (input);

  return (const char **) result;
}


static int
check_header (const struct header *header, const char **except)
{
  char line[BUFSIZ], command[sizeof fmt + strlen (header->name)
			     + 2 * strlen (CC)
			     + strlen (INC) + strlen (macrofile)];
  FILE *input;
  int result = 0;
  int found[header->nsyms];
  int i;

  memset (found, '\0', header->nsyms * sizeof (int));

  printf ("=== %s ===\n", header->name);
  sprintf (command, fmt, header->name, CC, INC, CC, macrofile);

  /* First see whether this subset is supported at all.  */
  if (header->subset != NULL)
    {
      sprintf (line, testfmt, header->subset, header->subset, CC, INC, CC,
	       macrofile);
      if (xsystem (line))
	{
	  printf ("!! not available\n");
	  return 0;
	}
    }

  if (xsystem (command))
    {
      puts ("system() returned nonzero");
      result = 1;
    }
  input = fopen (macrofile, "r");

  if (input == NULL)
    {
      printf ("Could not read %s: ", macrofile);
      perror (NULL);
      return 1;
    }

  while (fgets (line, sizeof line, input) != NULL)
    {
      const char **ignore;
      if (strlen (line) < 9 || line[7] != ' ')
	{ /* "#define A" */
	  printf ("Malformed input, expected '#define MACRO'\ngot '%s'\n",
		  line);
	  result = 1;
	  continue;
	}

      /* Find next char after the macro identifier; this can be either
	 a space or an open parenthesis.  */
      line[8 + strcspn (&line[8], " (")] = '\0';

      /* Now check whether it's one of the required macros.  */
      for (i = 0; i < header->nsyms; ++i)
	if (!strcmp (&line[8], header->syms[i]))
	  break;
      if (i < header->nsyms)
	{
	  found[i] = 1;
	  continue;
	}

      /* Symbols starting with "_" are ok.  */
      if (line[8] == '_')
	continue;

      /* Maybe one of the symbols which are always defined.  */
      for (ignore = except; *ignore != NULL; ++ignore)
	if (! strcmp (&line[8], *ignore))
	  break;
      if (*ignore != NULL)
	continue;

      /* Otherwise the symbol better should match one of the following.  */
      for (i = 0; i < header->nmaybe; ++i)
	if (fnmatch (header->maybe[i], &line[8], 0) == 0)
	  break;
      if (i < header->nmaybe)
	continue;

      printf ("*  invalid macro `%s'\n", &line[8]);
      result |= 1;
    }
  fclose (input);

  for (i = 0; i < header->nsyms; ++i)
    if (found[i] == 0)
      printf ("** macro `%s' not defined\n", header->syms[i]);

  return result;
}
