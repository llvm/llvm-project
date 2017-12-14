//===-- sanitizer_syscall_generic.inc ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Generic implementations of internal_syscall* and internal_iserror.
//
//===----------------------------------------------------------------------===//

#if SANITIZER_FREEBSD || SANITIZER_MAC || SANITIZER_NETBSD
# define SYSCALL(name) SYS_ ## name
#else
# define SYSCALL(name) __NR_ ## name
#endif

#if SANITIZER_NETBSD
// We use 3 kinds of internal_syscall's for different types of retval in order
// to address differences in calling conventions (e.g. registers to place the
// return value in).
//   - internal_syscall     for 32-bit length (int, pid_t)
//   - internal_syscall64   for 64-bit length (off_t)
//   - internal_syscall_ptr for pointer and (s)size_t
# define  internal_syscall      syscall
# define  internal_syscall64    __syscall
// Handle syscall renames manually
# define SYS_stat SYS___stat50
# define SYS_lstat SYS___lstat50
# define SYS_fstat SYS___fstat50
# define SYS_gettimeofday SYS___gettimeofday50
# define SYS_wait4 SYS___wait450
# define SYS_getdents SYS___getdents30
# define SYS_sigaltstack SYS___sigaltstack14
# define SYS_sigprocmask SYS___sigprocmask14
# define SYS_nanosleep SYS___nanosleep50
# define SYS_clock_gettime SYS___clock_gettime50
# if SANITIZER_WORDSIZE == 64
#  define internal_syscall_ptr  __syscall
# else
#  define internal_syscall_ptr  syscall
# endif
#elif defined(__x86_64__) && (SANITIZER_FREEBSD || SANITIZER_MAC)
# define internal_syscall __syscall
# define internal_syscall64 __syscall
# define internal_syscall_ptr __syscall
# else
# define internal_syscall syscall
# define internal_syscall64 syscall
# define internal_syscall_ptr syscall
#endif

bool internal_iserror(uptr retval, int *rverrno) {
  if (retval == (uptr)-1) {
    if (rverrno)
      *rverrno = errno;
    return true;
  } else {
    return false;
  }
}
