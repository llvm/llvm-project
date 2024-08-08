include "config/public_api.td"

include "spec/stdc.td"
include "spec/posix.td"
include "spec/linux.td"
include "spec/gnu_ext.td"
include "spec/bsd_ext.td"
include "spec/stdc_ext.td"
include "spec/llvm_libc_ext.td"
include "spec/llvm_libc_stdfix_ext.td"

def CTypeAPI : PublicAPI<"ctype.h"> {
}

def FCntlAPI : PublicAPI<"fcntl.h"> {
  let Types = [
    "mode_t",
    "off_t",
  ];
}

def IntTypesAPI : PublicAPI<"inttypes.h"> {
  let Types = ["imaxdiv_t"];
}

def MathAPI : PublicAPI<"math.h"> {
  let Types = ["double_t", "float_t", "float128"];
}

def FenvAPI: PublicAPI<"fenv.h"> {
  let Types = ["fenv_t", "fexcept_t"];
}

def StringAPI : PublicAPI<"string.h"> {
  let Types = ["size_t"];
}

def StdIOAPI : PublicAPI<"stdio.h"> {
  let Macros = [
    SimpleMacroDef<"stderr", "stderr">,
    SimpleMacroDef<"stdin", "stdin">,
    SimpleMacroDef<"stdout", "stdout">,
  ];
  let Types = [
    "FILE",
    "cookie_io_functions_t",
    "off_t",
    "size_t",
  ];
}

def StdlibAPI : PublicAPI<"stdlib.h"> {
  let Types = [
    "div_t",
    "ldiv_t",
    "lldiv_t",
    "size_t",
    "__bsearchcompare_t",
    "__qsortcompare_t",
    "__qsortrcompare_t",
    "__atexithandler_t",
  ];
}

def TimeAPI : PublicAPI<"time.h"> {
  let Types = [
    "clock_t",
    "time_t",
    "struct tm",
    "struct timespec",
    "struct timeval",
    "clockid_t",
  ];
}

def SchedAPI : PublicAPI<"sched.h"> {
  let Types = [
    "pid_t",
    "size_t",
    "cpu_set_t",
    "struct sched_param",
    // Needed according to posix standard
    "time_t",
    "struct timespec",
  ];
}

def SysMManAPI : PublicAPI<"sys/mman.h"> {
  let Types = ["off_t", "size_t", "mode_t"];
}

def SignalAPI : PublicAPI<"signal.h"> {
  let Types = [
    "sig_atomic_t",
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
      "pthread_condattr_t",
      "pthread_key_t",
      "pthread_mutex_t",
      "pthread_mutexattr_t",
      "pthread_once_t",
      "pthread_rwlockattr_t",
      "pthread_rwlock_t",
      "pthread_spinlock_t",
      "pthread_t",
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
  let Types = ["__exec_argv_t", "__exec_envp_t", "off_t", "pid_t", "size_t",
               "ssize_t", "uid_t", "__getoptargv_t"];
}

def WCharAPI : PublicAPI<"wchar.h"> {
  let Types = [
    "wchar_t",
    "wint_t",
    "size_t",
  ];
}

def UCharAPI : PublicAPI<"uchar.h"> {
  let Types = [
    "mbstate_t",
    "char8_t",
    "char16_t",
    "char32_t",
  ];
}

def SysRandomAPI : PublicAPI<"sys/random.h"> {
  let Types = ["size_t", "ssize_t"];
}

def SysSelectAPI : PublicAPI<"sys/select.h"> {
  let Types = ["fd_set", "sigset_t", "suseconds_t", "time_t", "struct timespec",
               "struct timeval"];
}

def SysSocketAPI : PublicAPI<"sys/socket.h"> {
  let Types = [
    "sa_family_t",
    "socklen_t",
    "struct sockaddr",
    "struct sockaddr_un",
  ];
}

def SysResourceAPI : PublicAPI<"sys/resource.h"> {
  let Types = ["rlim_t", "struct rlimit"];
}

def SysStatAPI : PublicAPI<"sys/stat.h"> {
  let Types = ["mode_t", "dev_t", "ino_t", "nlink_t", "uid_t", "gid_t", "off_t",
               "struct timespec", "struct timeval", "blksize_t", "blkcnt_t",
               "struct stat"];
}

def SysWaitAPI : PublicAPI<"sys/wait.h"> {
  let Types = ["pid_t", "struct rusage", "siginfo_t"];
}

def SysSendfileAPI : PublicAPI<"sys/sendfile.h"> {
  let Types = ["off_t", "size_t", "ssize_t"];
}

def SysTypesAPI : PublicAPI<"sys/types.h"> {
  let Types = [
    "blkcnt_t",
    "blksize_t",
    "clockid_t",
    "dev_t",
    "gid_t",
    "ino_t",
    "mode_t",
    "nlink_t",
    "off_t",
    "pid_t",
    "pthread_attr_t",
    "pthread_condattr_t",
    "pthread_key_t",
    "pthread_mutex_t",
    "pthread_mutexattr_t",
    "pthread_once_t",
    "pthread_rwlockattr_t",
    "pthread_rwlock_t",
    "pthread_t",
    "size_t",
    "ssize_t",
    "suseconds_t",
    "time_t",
    "uid_t"
  ];
}

def SysUtsNameAPI : PublicAPI<"sys/utsname.h"> {
  let Types = ["struct utsname"];
}

def SysEpollAPI : PublicAPI<"sys/epoll.h"> {
  let Types = ["struct epoll_event", "struct epoll_data", "sigset_t", "struct timespec"];
}

def SpawnAPI : PublicAPI<"spawn.h"> {
  let Types = ["mode_t", "pid_t", "posix_spawnattr_t", "posix_spawn_file_actions_t"];
}

def TermiosAPI : PublicAPI<"termios.h"> {
  let Types = ["cc_t", "pid_t", "speed_t", "struct termios", "tcflag_t"];
}

def SetJmpAPI : PublicAPI<"setjmp.h"> {
  let Types = ["jmp_buf"];
}

def SearchAPI : PublicAPI<"search.h"> {
  let Types = ["ACTION", "ENTRY", "struct hsearch_data"];
}

def SysStatvfsAPI : PublicAPI<"sys/statvfs.h"> {
  let Types = ["struct statvfs"];
}
