#ifndef _UNISTD_H
# include <posix/unistd.h>

# ifndef _ISOMAC

libc_hidden_proto (_exit, __noreturn__)
#  ifndef NO_RTLD_HIDDEN
rtld_hidden_proto (_exit, __noreturn__)
#  endif
libc_hidden_proto (alarm)
extern size_t __confstr (int name, char *buf, size_t len);
libc_hidden_proto (__confstr)
libc_hidden_proto (confstr)
libc_hidden_proto (execl)
libc_hidden_proto (execle)
libc_hidden_proto (execlp)
libc_hidden_proto (execvp)
libc_hidden_proto (getpid)
libc_hidden_proto (getsid)
libc_hidden_proto (getdomainname)
extern __typeof (getlogin_r) __getlogin_r  __nonnull ((1));
libc_hidden_proto (__getlogin_r)
libc_hidden_proto (getlogin_r)
libc_hidden_proto (seteuid)
libc_hidden_proto (setegid)
libc_hidden_proto (tcgetpgrp)
libc_hidden_proto (readlinkat)
libc_hidden_proto (fsync)
libc_hidden_proto (fdatasync)

/* Now define the internal interfaces.  */
extern int __access (const char *__name, int __type);
libc_hidden_proto (__access)
extern int __euidaccess (const char *__name, int __type);
extern int __faccessat (int __fd, const char *__file, int __type, int __flag);
extern int __faccessat_noerrno (int __fd, const char *__file, int __type,
			        int __flag);
extern __off64_t __lseek64 (int __fd, __off64_t __offset, int __whence);
extern __off_t __lseek (int __fd, __off_t __offset, int __whence);
libc_hidden_proto (__lseek)
extern __off_t __libc_lseek (int __fd, __off_t __offset, int __whence);
extern __off64_t __libc_lseek64 (int __fd, __off64_t __offset, int __whence);
extern ssize_t __pread (int __fd, void *__buf, size_t __nbytes,
			__off_t __offset);
libc_hidden_proto (__pread);
extern ssize_t __libc_pread (int __fd, void *__buf, size_t __nbytes,
			     __off_t __offset);
extern ssize_t __pread64 (int __fd, void *__buf, size_t __nbytes,
			  __off64_t __offset);
libc_hidden_proto (__pread64);
extern ssize_t __libc_pread64 (int __fd, void *__buf, size_t __nbytes,
			       __off64_t __offset);
extern ssize_t __pwrite (int __fd, const void *__buf, size_t __n,
			 __off_t __offset);
libc_hidden_proto (__pwrite)
extern ssize_t __libc_pwrite (int __fd, const void *__buf, size_t __n,
			      __off_t __offset);
extern ssize_t __pwrite64 (int __fd, const void *__buf, size_t __n,
			   __off64_t __offset);
libc_hidden_proto (__pwrite64)
extern ssize_t __libc_pwrite64 (int __fd, const void *__buf, size_t __n,
				__off64_t __offset);
extern ssize_t __libc_read (int __fd, void *__buf, size_t __n);
libc_hidden_proto (__libc_read)
libc_hidden_proto (read)
extern ssize_t __libc_write (int __fd, const void *__buf, size_t __n);
libc_hidden_proto (__libc_write)
libc_hidden_proto (write)
extern int __pipe (int __pipedes[2]);
libc_hidden_proto (__pipe)
extern int __pipe2 (int __pipedes[2], int __flags) attribute_hidden;
extern unsigned int __sleep (unsigned int __seconds) attribute_hidden;
extern int __chown (const char *__file,
		    __uid_t __owner, __gid_t __group);
libc_hidden_proto (__chown)
extern int __fchown (int __fd,
		     __uid_t __owner, __gid_t __group);
extern int __lchown (const char *__file, __uid_t __owner,
		     __gid_t __group);
extern int __chdir (const char *__path) attribute_hidden;
extern int __fchdir (int __fd) attribute_hidden;
extern char *__getcwd (char *__buf, size_t __size);
libc_hidden_proto (__getcwd)
extern int __rmdir (const char *__path) attribute_hidden;
extern int __execvpe (const char *file, char *const argv[],
		      char *const envp[]) attribute_hidden;
extern int __execvpex (const char *file, char *const argv[],
		       char *const envp[]) attribute_hidden;

/* Get the canonical absolute name of the named directory, and put it in SIZE
   bytes of BUF.  Returns NULL if the directory couldn't be determined or
   SIZE was too small.  If successful, returns BUF.  In GNU, if BUF is
   NULL, an array is allocated with `malloc'; the array is SIZE bytes long,
   unless SIZE <= 0, in which case it is as big as necessary.  */

char *__canonicalize_directory_name_internal (const char *__thisdir,
					      char *__buf,
					      size_t __size) attribute_hidden;

extern int __dup (int __fd);
libc_hidden_proto (__dup)
extern int __dup2 (int __fd, int __fd2);
libc_hidden_proto (__dup2)
extern int __dup3 (int __fd, int __fd2, int flags);
libc_hidden_proto (__dup3)
extern int __execve (const char *__path, char *const __argv[],
		     char *const __envp[]) attribute_hidden;
extern int __execveat (int dirfd, const char *__path, char *const __argv[],
		       char *const __envp[], int flags) attribute_hidden;
extern long int __pathconf (const char *__path, int __name);
extern long int __fpathconf (int __fd, int __name);
extern long int __sysconf (int __name);
libc_hidden_proto (__sysconf)
extern __pid_t __getpid (void);
libc_hidden_proto (__getpid)
extern __pid_t __getppid (void);
extern __pid_t __setsid (void) attribute_hidden;
extern __uid_t __getuid (void) attribute_hidden;
extern __uid_t __geteuid (void) attribute_hidden;
extern __gid_t __getgid (void) attribute_hidden;
extern __gid_t __getegid (void) attribute_hidden;
extern int __getgroups (int __size, __gid_t __list[]) attribute_hidden;
libc_hidden_proto (__getpgid)
extern int __group_member (__gid_t __gid) attribute_hidden;
extern int __setuid (__uid_t __uid);
extern int __setreuid (__uid_t __ruid, __uid_t __euid);
extern int __setgid (__gid_t __gid);
extern int __setpgid (__pid_t __pid, __pid_t __pgid);
libc_hidden_proto (__setpgid)
extern int __setregid (__gid_t __rgid, __gid_t __egid);
extern int __getresuid (__uid_t *__ruid, __uid_t *__euid, __uid_t *__suid);
extern int __getresgid (__gid_t *__rgid, __gid_t *__egid, __gid_t *__sgid);
extern int __setresuid (__uid_t __ruid, __uid_t __euid, __uid_t __suid);
extern int __setresgid (__gid_t __rgid, __gid_t __egid, __gid_t __sgid);
libc_hidden_proto (__getresuid)
libc_hidden_proto (__getresgid)
libc_hidden_proto (__setresuid)
libc_hidden_proto (__setresgid)
extern __pid_t __vfork (void);
libc_hidden_proto (__vfork)
extern int __ttyname_r (int __fd, char *__buf, size_t __buflen);
libc_hidden_proto (__ttyname_r)
extern __pid_t _Fork (void);
libc_hidden_proto (_Fork);
extern int __isatty (int __fd) attribute_hidden;
extern int __link (const char *__from, const char *__to);
extern int __symlink (const char *__from, const char *__to);
extern ssize_t __readlink (const char *__path, char *__buf, size_t __len)
     attribute_hidden;
extern int __unlink (const char *__name) attribute_hidden;
extern int __gethostname (char *__name, size_t __len) attribute_hidden;
extern int __revoke (const char *__file);
extern int __profil (unsigned short int *__sample_buffer, size_t __size,
		     size_t __offset, unsigned int __scale)
     attribute_hidden;
extern int __getdtablesize (void) attribute_hidden;
extern int __brk (void *__addr) attribute_hidden;
extern int __close (int __fd);
libc_hidden_proto (__close)
extern int __libc_close (int __fd);
extern _Bool __closefrom_fallback (int __lowfd, _Bool) attribute_hidden;
extern ssize_t __read (int __fd, void *__buf, size_t __nbytes);
libc_hidden_proto (__read)
extern ssize_t __write (int __fd, const void *__buf, size_t __n);
libc_hidden_proto (__write)
extern __pid_t __fork (void);
libc_hidden_proto (__fork)
extern int __getpagesize (void) __attribute__ ((__const__));
libc_hidden_proto (__getpagesize)
extern int __ftruncate (int __fd, __off_t __length) attribute_hidden;
extern int __ftruncate64 (int __fd, __off64_t __length) attribute_hidden;
extern int __truncate (const char *path, __off_t __length);
extern void *__sbrk (intptr_t __delta);
libc_hidden_proto (__sbrk)


/* This variable is set nonzero at startup if the process's effective
   IDs differ from its real IDs, or it is otherwise indicated that
   extra security should be used.  When this is set the dynamic linker
   and some functions contained in the C library ignore various
   environment variables that normally affect them.  */
extern int __libc_enable_secure attribute_relro;
extern int __libc_enable_secure_decided;
rtld_hidden_proto (__libc_enable_secure)


/* Various internal function.  */
extern void __libc_check_standard_fds (void) attribute_hidden;


/* Internal name for fork function.  */
extern __pid_t __libc_fork (void);

/* Suspend the process until a signal arrives.
   This always returns -1 and sets `errno' to EINTR.  */
extern int __libc_pause (void);

extern int __getlogin_r_loginuid (char *name, size_t namesize)
     attribute_hidden;

#  if IS_IN (rtld)
#   include <dl-unistd.h>
#  endif

# endif
#endif
