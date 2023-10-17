// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:DisplayLoadedSummaries=true \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple i686-unknown-linux -verify

// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:DisplayLoadedSummaries=true \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple i686-unknown-linux 2>&1 | FileCheck %s

// CHECK: Loaded summary for: FILE *fopen(const char *restrict pathname, const char *restrict mode)
// CHECK: Loaded summary for: FILE *tmpfile(void)
// CHECK: Loaded summary for: FILE *freopen(const char *restrict pathname, const char *restrict mode, FILE *restrict stream)
// CHECK: Loaded summary for: int fclose(FILE *stream)
// CHECK: Loaded summary for: int fseek(FILE *stream, long offset, int whence)
// CHECK: Loaded summary for: int fileno(FILE *stream)
// CHECK: Loaded summary for: long a64l(const char *str64)
// CHECK: Loaded summary for: char *l64a(long value)
// CHECK: Loaded summary for: int open(const char *path, int oflag, ...)
// CHECK: Loaded summary for: int openat(int fd, const char *path, int oflag, ...)
// CHECK: Loaded summary for: int access(const char *pathname, int amode)
// CHECK: Loaded summary for: int faccessat(int dirfd, const char *pathname, int mode, int flags)
// CHECK: Loaded summary for: int dup(int fildes)
// CHECK: Loaded summary for: int dup2(int fildes1, int filedes2)
// CHECK: Loaded summary for: int fdatasync(int fildes)
// CHECK: Loaded summary for: int fnmatch(const char *pattern, const char *string, int flags)
// CHECK: Loaded summary for: int fsync(int fildes)
// CHECK: Loaded summary for: int truncate(const char *path, off_t length)
// CHECK: Loaded summary for: int symlink(const char *oldpath, const char *newpath)
// CHECK: Loaded summary for: int symlinkat(const char *oldpath, int newdirfd, const char *newpath)
// CHECK: Loaded summary for: int lockf(int fd, int cmd, off_t len)
// CHECK: Loaded summary for: int creat(const char *pathname, mode_t mode)
// CHECK: Loaded summary for: unsigned int sleep(unsigned int seconds)
// CHECK: Loaded summary for: int dirfd(DIR *dirp)
// CHECK: Loaded summary for: unsigned int alarm(unsigned int seconds)
// CHECK: Loaded summary for: int closedir(DIR *dir)
// CHECK: Loaded summary for: char *strdup(const char *s)
// CHECK: Loaded summary for: char *strndup(const char *s, size_t n)
// CHECK: Loaded summary for: int mkstemp(char *template)
// CHECK: Loaded summary for: char *mkdtemp(char *template)
// CHECK: Loaded summary for: char *getcwd(char *buf, size_t size)
// CHECK: Loaded summary for: int mkdir(const char *pathname, mode_t mode)
// CHECK: Loaded summary for: int mkdirat(int dirfd, const char *pathname, mode_t mode)
// CHECK: Loaded summary for: int mknod(const char *pathname, mode_t mode, dev_t dev)
// CHECK: Loaded summary for: int mknodat(int dirfd, const char *pathname, mode_t mode, dev_t dev)
// CHECK: Loaded summary for: int chmod(const char *path, mode_t mode)
// CHECK: Loaded summary for: int fchmodat(int dirfd, const char *pathname, mode_t mode, int flags)
// CHECK: Loaded summary for: int fchmod(int fildes, mode_t mode)
// CHECK: Loaded summary for: int fchownat(int dirfd, const char *pathname, uid_t owner, gid_t group, int flags)
// CHECK: Loaded summary for: int chown(const char *path, uid_t owner, gid_t group)
// CHECK: Loaded summary for: int lchown(const char *path, uid_t owner, gid_t group)
// CHECK: Loaded summary for: int fchown(int fildes, uid_t owner, gid_t group)
// CHECK: Loaded summary for: int rmdir(const char *pathname)
// CHECK: Loaded summary for: int chdir(const char *path)
// CHECK: Loaded summary for: int link(const char *oldpath, const char *newpath)
// CHECK: Loaded summary for: int linkat(int fd1, const char *path1, int fd2, const char *path2, int flag)
// CHECK: Loaded summary for: int unlink(const char *pathname)
// CHECK: Loaded summary for: int unlinkat(int fd, const char *path, int flag)
// CHECK: Loaded summary for: int fstat(int fd, struct stat *statbuf)
// CHECK: Loaded summary for: int stat(const char *restrict path, struct stat *restrict buf)
// CHECK: Loaded summary for: int lstat(const char *restrict path, struct stat *restrict buf)
// CHECK: Loaded summary for: int fstatat(int fd, const char *restrict path, struct stat *restrict buf, int flag)
// CHECK: Loaded summary for: DIR *opendir(const char *name)
// CHECK: Loaded summary for: DIR *fdopendir(int fd)
// CHECK: Loaded summary for: int isatty(int fildes)
// CHECK: Loaded summary for: FILE *popen(const char *command, const char *type)
// CHECK: Loaded summary for: int pclose(FILE *stream)
// CHECK: Loaded summary for: int close(int fildes)
// CHECK: Loaded summary for: long fpathconf(int fildes, int name)
// CHECK: Loaded summary for: long pathconf(const char *path, int name)
// CHECK: Loaded summary for: FILE *fdopen(int fd, const char *mode)
// CHECK: Loaded summary for: void rewinddir(DIR *dir)
// CHECK: Loaded summary for: void seekdir(DIR *dirp, long loc)
// CHECK: Loaded summary for: int rand_r(unsigned int *seedp)
// CHECK: Loaded summary for: int fseeko(FILE *stream, off_t offset, int whence)
// CHECK: Loaded summary for: off_t ftello(FILE *stream)
// CHECK: Loaded summary for: void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset)
// CHECK: Loaded summary for: void *mmap64(void *addr, size_t length, int prot, int flags, int fd, off64_t offset)
// CHECK: Loaded summary for: int pipe(int fildes[2])
// CHECK: Loaded summary for: off_t lseek(int fildes, off_t offset, int whence)
// CHECK: Loaded summary for: ssize_t readlink(const char *restrict path, char *restrict buf, size_t bufsize)
// CHECK: Loaded summary for: ssize_t readlinkat(int fd, const char *restrict path, char *restrict buf, size_t bufsize)
// CHECK: Loaded summary for: int renameat(int olddirfd, const char *oldpath, int newdirfd, const char *newpath)
// CHECK: Loaded summary for: char *realpath(const char *restrict file_name, char *restrict resolved_name)
// CHECK: Loaded summary for: int execv(const char *path, char *const argv[])
// CHECK: Loaded summary for: int execvp(const char *file, char *const argv[])
// CHECK: Loaded summary for: int getopt(int argc, char *const argv[], const char *optstring)
// CHECK: Loaded summary for: int socket(int domain, int type, int protocol)
// CHECK: Loaded summary for: int accept(int socket, __SOCKADDR_ARG address, socklen_t *restrict address_len)
// CHECK: Loaded summary for: int bind(int socket, __CONST_SOCKADDR_ARG address, socklen_t address_len)
// CHECK: Loaded summary for: int getpeername(int socket, __SOCKADDR_ARG address, socklen_t *restrict address_len)
// CHECK: Loaded summary for: int getsockname(int socket, __SOCKADDR_ARG address, socklen_t *restrict address_len)
// CHECK: Loaded summary for: int connect(int socket, __CONST_SOCKADDR_ARG address, socklen_t address_len)
// CHECK: Loaded summary for: ssize_t recvfrom(int socket, void *restrict buffer, size_t length, int flags, __SOCKADDR_ARG address, socklen_t *restrict address_len)
// CHECK: Loaded summary for: ssize_t sendto(int socket, const void *message, size_t length, int flags, __CONST_SOCKADDR_ARG dest_addr, socklen_t dest_len)
// CHECK: Loaded summary for: int listen(int sockfd, int backlog)
// CHECK: Loaded summary for: ssize_t recv(int sockfd, void *buf, size_t len, int flags)
// CHECK: Loaded summary for: ssize_t recvmsg(int sockfd, struct msghdr *msg, int flags)
// CHECK: Loaded summary for: ssize_t sendmsg(int sockfd, const struct msghdr *msg, int flags)
// CHECK: Loaded summary for: int setsockopt(int socket, int level, int option_name, const void *option_value, socklen_t option_len)
// CHECK: Loaded summary for: int getsockopt(int socket, int level, int option_name, void *restrict option_value, socklen_t *restrict option_len)
// CHECK: Loaded summary for: ssize_t send(int sockfd, const void *buf, size_t len, int flags)
// CHECK: Loaded summary for: int socketpair(int domain, int type, int protocol, int sv[2])
// CHECK: Loaded summary for: int shutdown(int socket, int how)
// CHECK: Loaded summary for: int getnameinfo(const struct sockaddr *restrict sa, socklen_t salen, char *restrict node, socklen_t nodelen, char *restrict service, socklen_t servicelen, int flags)
// CHECK: Loaded summary for: int utime(const char *filename, struct utimbuf *buf)
// CHECK: Loaded summary for: int futimens(int fd, const struct timespec times[2])
// CHECK: Loaded summary for: int utimensat(int dirfd, const char *pathname, const struct timespec times[2], int flags)
// CHECK: Loaded summary for: int utimes(const char *filename, const struct timeval times[2])
// CHECK: Loaded summary for: int nanosleep(const struct timespec *rqtp, struct timespec *rmtp)
// CHECK: Loaded summary for: struct tm *localtime(const time_t *tp)
// CHECK: Loaded summary for: struct tm *localtime_r(const time_t *restrict timer, struct tm *restrict result)
// CHECK: Loaded summary for: char *asctime_r(const struct tm *restrict tm, char *restrict buf)
// CHECK: Loaded summary for: char *ctime_r(const time_t *timep, char *buf)
// CHECK: Loaded summary for: struct tm *gmtime_r(const time_t *restrict timer, struct tm *restrict result)
// CHECK: Loaded summary for: struct tm *gmtime(const time_t *tp)
// CHECK: Loaded summary for: int clock_gettime(clockid_t clock_id, struct timespec *tp)
// CHECK: Loaded summary for: int getitimer(int which, struct itimerval *curr_value)
// CHECK: Loaded summary for: int pthread_cond_signal(pthread_cond_t *cond)
// CHECK: Loaded summary for: int pthread_cond_broadcast(pthread_cond_t *cond)
// CHECK: Loaded summary for: int pthread_create(pthread_t *restrict thread, const pthread_attr_t *restrict attr, void *(*start_routine)(void *), void *restrict arg)
// CHECK: Loaded summary for: int pthread_attr_destroy(pthread_attr_t *attr)
// CHECK: Loaded summary for: int pthread_attr_init(pthread_attr_t *attr)
// CHECK: Loaded summary for: int pthread_attr_getstacksize(const pthread_attr_t *restrict attr, size_t *restrict stacksize)
// CHECK: Loaded summary for: int pthread_attr_getguardsize(const pthread_attr_t *restrict attr, size_t *restrict guardsize)
// CHECK: Loaded summary for: int pthread_attr_setstacksize(pthread_attr_t *attr, size_t stacksize)
// CHECK: Loaded summary for: int pthread_attr_setguardsize(pthread_attr_t *attr, size_t guardsize)
// CHECK: Loaded summary for: int pthread_mutex_init(pthread_mutex_t *restrict mutex, const pthread_mutexattr_t *restrict attr)
// CHECK: Loaded summary for: int pthread_mutex_destroy(pthread_mutex_t *mutex)
// CHECK: Loaded summary for: int pthread_mutex_lock(pthread_mutex_t *mutex)
// CHECK: Loaded summary for: int pthread_mutex_trylock(pthread_mutex_t *mutex)
// CHECK: Loaded summary for: int pthread_mutex_unlock(pthread_mutex_t *mutex)

#include "Inputs/std-c-library-functions-POSIX.h"

void clang_analyzer_eval(int);

void test_open(void) {
  open(0, 0); // \
  // expected-warning{{The 1st argument to 'open' is NULL but should not be NULL}}
}

void test_open_additional_arg(void) {
  open(0, 0, 0); // \
  // expected-warning{{The 1st argument to 'open' is NULL but should not be NULL}}
}

void test_recvfrom(int socket, void *restrict buffer, size_t length, int flags,
                   struct sockaddr *restrict address,
                   socklen_t *restrict address_len) {
  ssize_t Ret = recvfrom(socket, buffer, length, flags, address, address_len);
  if (Ret == 0)
    clang_analyzer_eval(length == 0); // expected-warning{{TRUE}}
  if (Ret > 0)
    clang_analyzer_eval(length > 0); // expected-warning{{TRUE}}
  if (Ret == -1)
    clang_analyzer_eval(length == 0); // expected-warning{{UNKNOWN}}
}

void test_sendto(int socket, const void *message, size_t length, int flags,
                 const struct sockaddr *dest_addr, socklen_t dest_len) {
  ssize_t Ret = sendto(socket, message, length, flags, dest_addr, dest_len);
  if (Ret == 0)
    clang_analyzer_eval(length == 0); // expected-warning{{TRUE}}
  if (Ret > 0)
    clang_analyzer_eval(length > 0); // expected-warning{{TRUE}}
  if (Ret == -1)
    clang_analyzer_eval(length == 0); // expected-warning{{UNKNOWN}}
}

void test_recv(int sockfd, void *buf, size_t len, int flags) {
  ssize_t Ret = recv(sockfd, buf, len, flags);
  if (Ret == 0)
    clang_analyzer_eval(len == 0); // expected-warning{{TRUE}}
  if (Ret > 0)
    clang_analyzer_eval(len > 0); // expected-warning{{TRUE}}
  if (Ret == -1)
    clang_analyzer_eval(len == 0); // expected-warning{{UNKNOWN}}
}

void test_send(int sockfd, void *buf, size_t len, int flags) {
  ssize_t Ret = send(sockfd, buf, len, flags);
  if (Ret == 0)
    clang_analyzer_eval(len == 0); // expected-warning{{TRUE}}
  if (Ret > 0)
    clang_analyzer_eval(len > 0); // expected-warning{{TRUE}}
  if (Ret == -1)
    clang_analyzer_eval(len == 0); // expected-warning{{UNKNOWN}}
}

void test_recvmsg(int sockfd, struct msghdr *msg, int flags) {
  ssize_t Ret = recvmsg(sockfd, msg, flags);
  clang_analyzer_eval(Ret != 0); // expected-warning{{TRUE}}
}

void test_sendmsg(int sockfd, const struct msghdr *msg, int flags) {
  ssize_t Ret = sendmsg(sockfd, msg, flags);
  clang_analyzer_eval(Ret != 0); // expected-warning{{TRUE}}
}
