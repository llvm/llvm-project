include "config/public_api.td"

include "spec/bsd_ext.td"
include "spec/gnu_ext.td"
include "spec/linux.td"
include "spec/llvm_libc_ext.td"
include "spec/posix.td"
include "spec/stdc.td"

def AssertMacro : MacroDef<"assert"> {
  let Defn = [{
    #undef assert

    #ifdef NDEBUG
    #define assert(e) (void)0
    #else

    #ifdef __cplusplus
    extern "C"
    #endif
    _Noreturn void __assert_fail(const char *, const char *, unsigned, const char *);

    #define assert(e)  \
      ((e) ? (void)0 : __assert_fail(#e, __FILE__, __LINE__, __PRETTY_FUNCTION__))

    #endif
  }];
}

def StaticAssertMacro : MacroDef<"static_assert"> {
  let Defn = [{
    #ifndef __cplusplus
    #undef static_assert
    #define static_assert _Static_assert
    #endif
  }];
}

def NullMacro : MacroDef<"NULL"> {
  let Defn = [{
    #define __need_NULL
    #include <stddef.h>
  }];
}

def ErrnoMacro : MacroDef<"errno"> {
  let Defn = [{
    extern _Thread_local int __llvmlibc_errno;
    #define errno __llvmlibc_errno
  }];
}

def AssertAPI : PublicAPI<"assert.h"> {
  let Macros = [
    AssertMacro,
    StaticAssertMacro,
  ];
}

def CTypeAPI : PublicAPI<"ctype.h"> {
}

def FCntlAPI : PublicAPI<"fcntl.h"> {
  let Types = ["mode_t"];
}

def IntTypesAPI : PublicAPI<"inttypes.h"> {
  let Types = ["imaxdiv_t"];
}

def MathErrHandlingMacro : MacroDef<"math_errhandling"> {
  let Defn = [{
    #ifndef math_errhandling
    #ifdef __FAST_MATH__
    #define math_errhandling 0
    #elif defined __NO_MATH_ERRNO__
    #define math_errhandling (MATH_ERREXCEPT)
    #else
    #define math_errhandling (MATH_ERRNO | MATH_ERREXCEPT)
    #endif
    #endif // math_errhandling not defined
  }];
}

def IsFiniteMacro : MacroDef<"isfinite"> {
  let Defn = [{
    #define isfinite(x) __builtin_isfinite(x)
  }];
}

def IsInfMacro : MacroDef<"isinf"> {
  let Defn = [{
    #define isinf(x) __builtin_isinf(x)
  }];
}

def IsNanMacro : MacroDef<"isnan"> {
  let Defn = [{
    #define isnan(x) __builtin_isnan(x)
  }];
}

def MathAPI : PublicAPI<"math.h"> {
  let Macros = [
    SimpleMacroDef<"MATH_ERRNO", "1">,
    SimpleMacroDef<"MATH_ERREXCEPT", "2">,
    MathErrHandlingMacro,

    SimpleMacroDef<"INFINITY", "__builtin_inff()">,
    SimpleMacroDef<"NAN", "__builtin_nanf(\"\")">,

    SimpleMacroDef<"FP_ILOGB0", "(-__INT_MAX__ - 1)">, // INT_MIN
    SimpleMacroDef<"FP_ILOGBNAN", "__INT_MAX__">,

    IsFiniteMacro,
    IsInfMacro,
    IsNanMacro,
  ];
  let Types = ["double_t", "float_t"];
}

def FenvAPI: PublicAPI<"fenv.h"> {
  let Macros = [
    SimpleMacroDef<"FE_DIVBYZERO", "1">,
    SimpleMacroDef<"FE_INEXACT", "2">,
    SimpleMacroDef<"FE_INVALID", "4">,
    SimpleMacroDef<"FE_OVERFLOW", "8">,
    SimpleMacroDef<"FE_UNDERFLOW", "16">,
    SimpleMacroDef<"FE_ALL_EXCEPT", "(FE_DIVBYZERO|FE_INEXACT|FE_INVALID|FE_OVERFLOW|FE_UNDERFLOW)">,

    SimpleMacroDef<"FE_DOWNWARD", "1">,
    SimpleMacroDef<"FE_TONEAREST", "2">,
    SimpleMacroDef<"FE_TOWARDZERO", "4">,
    SimpleMacroDef<"FE_UPWARD", "8">,

    SimpleMacroDef<"FE_DFL_ENV", "((fenv_t *)-1)">,
  ];
  let Types = ["fenv_t", "fexcept_t"];
}

def StringAPI : PublicAPI<"string.h"> {
  let Types = ["size_t"];

  let Macros = [
    NullMacro,
  ];
}

def StdIOAPI : PublicAPI<"stdio.h"> {
  let Macros = [
    SimpleMacroDef<"stderr", "stderr">,
    SimpleMacroDef<"stdin", "stdin">,
    SimpleMacroDef<"stdout", "stdout">,
    SimpleMacroDef<"_IOFBF", "0">,
    SimpleMacroDef<"_IOLBF", "1">,
    SimpleMacroDef<"_IONBF", "2">,
    SimpleMacroDef<"EOF", "-1">,
  ];
  let Types = ["size_t", "FILE", "cookie_io_functions_t"];
}

def StdlibAPI : PublicAPI<"stdlib.h"> {
  let Types = [
    "div_t",
    "ldiv_t",
    "lldiv_t",
    "size_t",
    "__bsearchcompare_t",
    "__qsortcompare_t",
    "__atexithandler_t",
  ];
}

def TimeAPI : PublicAPI<"time.h"> {
  let Types = ["clock_t", "time_t", "struct tm", "struct timespec", "clockid_t",];
}

def ErrnoAPI : PublicAPI<"errno.h"> {
  let Macros = [
    ErrnoMacro,
    // We largely depend on linux/errno.h to give us the
    // various error macro definitions. However, some libc
    // implementations have chosen to provide definitions
    // for some of the error macros to account for the ones
    // missing in linux/errno.h. There is no harm in doing
    // the same here if we define the macros only when they
    // are not already defined.
    MacroDefineIfNot<"ENOTSUP", "EOPNOTSUPP">,
    MacroDefineIfNot<"ECANCELED", "125">,
    MacroDefineIfNot<"EOWNERDEAD", "130">,
    MacroDefineIfNot<"ENOTRECOVERABLE", "131">,
    MacroDefineIfNot<"ERFKILL", "132">,
    MacroDefineIfNot<"EHWPOISON", "133">,
  ];
}

def SchedAPI : PublicAPI<"sched.h"> {
  let Types = ["pid_t", "size_t", "cpu_set_t"];
}

def SysMManAPI : PublicAPI<"sys/mman.h"> {
  let Types = ["off_t", "size_t"];
  let Macros = [
  ];

}

def SignalAPI : PublicAPI<"signal.h"> {
  let Types = [
    "sigset_t",
    "struct sigaction",
    "union sigval",
    "siginfo_t",
    "stack_t",
    "pid_t",
  ];
}

def ThreadsAPI : PublicAPI<"threads.h"> {
  let Macros = [
    SimpleMacroDef<"ONCE_FLAG_INIT", "{0}">,
  ];

  let Types = [
    "__call_once_func_t",
    "once_flag",
    "cnd_t",
    "mtx_t",
    "thrd_t",
    "thrd_start_t",
    "tss_t",
    "tss_dtor_t",
  ];

  let Enumerations = [
    "mtx_plain",
    "mtx_recursive",
    "mtx_timed",
    "thrd_timedout",
    "thrd_success",
    "thrd_busy",
    "thrd_error",
    "thrd_nomem",
  ];
}

def PThreadAPI : PublicAPI<"pthread.h"> {
  let Types = [
      "__atfork_callback_t",
      "__pthread_once_func_t",
      "__pthread_start_t",
      "__pthread_tss_dtor_t",
      "pthread_attr_t",
      "pthread_mutex_t",
      "pthread_mutexattr_t",
      "pthread_t",
      "pthread_key_t",
      "pthread_once_t",
  ];
}

def DirentAPI : PublicAPI<"dirent.h"> {
  let Types = [
    "ino_t",
    "DIR",
    "struct dirent",
  ];
}

def UniStdAPI : PublicAPI<"unistd.h"> {
  let Types = ["__exec_argv_t", "__exec_envp_t", "off_t", "pid_t", "size_t", "ssize_t", "uid_t"];
}

def SysRandomAPI : PublicAPI<"sys/random.h"> {
  let Types = ["size_t", "ssize_t"];
}

def SysSelectAPI : PublicAPI<"sys/select.h"> {
  let Types = ["fd_set", "sigset_t", "suseconds_t", "time_t", "struct timespec", "struct timeval"];
}

def SysResourceAPI : PublicAPI<"sys/resource.h"> {
  let Types = ["rlim_t", "struct rlimit"];
}

def SysStatAPI : PublicAPI<"sys/stat.h"> {
  let Types = ["mode_t", "dev_t", "ino_t", "nlink_t", "uid_t", "gid_t", "off_t",
               "struct timespec", "blksize_t", "blkcnt_t", "struct stat"];
}

def SysWaitAPI : PublicAPI<"sys/wait.h"> {
  let Types = ["pid_t", "struct rusage"];
}

def SysSendfileAPI : PublicAPI<"sys/sendfile.h"> {
  let Types = ["off_t", "size_t", "ssize_t"];
}

def SysTypesAPI : PublicAPI<"sys/types.h"> {
  let Types = ["blkcnt_t", "blksize_t", "clockid_t", "dev_t", "gid_t", "ino_t",
               "mode_t", "nlink_t", "off_t", "pid_t", "pthread_attr_t", "pthread_key_t",
               "pthread_mutex_t", "pthread_mutexattr_t", "pthread_once_t", "pthread_t",
               "size_t", "ssize_t", "suseconds_t", "time_t", "uid_t"];
}

def SysUtsNameAPI : PublicAPI<"sys/utsname.h"> {
  let Types = ["struct utsname"];
}

def SpawnAPI : PublicAPI<"spawn.h"> {
  let Types = ["mode_t", "pid_t", "posix_spawnattr_t", "posix_spawn_file_actions_t"];
}

def TermiosAPI : PublicAPI<"termios.h"> {
  let Types = ["cc_t", "pid_t", "speed_t", "struct termios", "tcflag_t"];
}
