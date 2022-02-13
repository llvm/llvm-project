// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-config apiModeling.StdCLibraryFunctions:ModelPOSIX=true \
// RUN:   -analyzer-config apiModeling.StdCLibraryFunctions:DisplayLoadedSummaries=true \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple i686-unknown-linux 2>&1 | FileCheck %s

// CHECK: Loaded summary for: long a64l(const char *str64)
// CHECK: Loaded summary for: char *l64a(long value)
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
// CHECK: Loaded summary for: int fileno(FILE *stream)
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

long a64l(const char *str64);
char *l64a(long value);
int access(const char *pathname, int amode);
int faccessat(int dirfd, const char *pathname, int mode, int flags);
int dup(int fildes);
int dup2(int fildes1, int filedes2);
int fdatasync(int fildes);
int fnmatch(const char *pattern, const char *string, int flags);
int fsync(int fildes);
typedef unsigned long off_t;
int truncate(const char *path, off_t length);
int symlink(const char *oldpath, const char *newpath);
int symlinkat(const char *oldpath, int newdirfd, const char *newpath);
int lockf(int fd, int cmd, off_t len);
typedef unsigned mode_t;
int creat(const char *pathname, mode_t mode);
unsigned int sleep(unsigned int seconds);
typedef struct {
  int a;
} DIR;
int dirfd(DIR *dirp);
unsigned int alarm(unsigned int seconds);
int closedir(DIR *dir);
char *strdup(const char *s);
typedef typeof(sizeof(int)) size_t;
char *strndup(const char *s, size_t n);
/*FIXME How to define wchar_t in the test?*/
/*typedef __wchar_t wchar_t;*/
/*wchar_t *wcsdup(const wchar_t *s);*/
int mkstemp(char *template);
char *mkdtemp(char *template);
char *getcwd(char *buf, size_t size);
int mkdir(const char *pathname, mode_t mode);
int mkdirat(int dirfd, const char *pathname, mode_t mode);
typedef int dev_t;
int mknod(const char *pathname, mode_t mode, dev_t dev);
int mknodat(int dirfd, const char *pathname, mode_t mode, dev_t dev);
int chmod(const char *path, mode_t mode);
int fchmodat(int dirfd, const char *pathname, mode_t mode, int flags);
int fchmod(int fildes, mode_t mode);
typedef int uid_t;
typedef int gid_t;
int fchownat(int dirfd, const char *pathname, uid_t owner, gid_t group, int flags);
int chown(const char *path, uid_t owner, gid_t group);
int lchown(const char *path, uid_t owner, gid_t group);
int fchown(int fildes, uid_t owner, gid_t group);
int rmdir(const char *pathname);
int chdir(const char *path);
int link(const char *oldpath, const char *newpath);
int linkat(int fd1, const char *path1, int fd2, const char *path2, int flag);
int unlink(const char *pathname);
int unlinkat(int fd, const char *path, int flag);
struct stat;
int fstat(int fd, struct stat *statbuf);
int stat(const char *restrict path, struct stat *restrict buf);
int lstat(const char *restrict path, struct stat *restrict buf);
int fstatat(int fd, const char *restrict path, struct stat *restrict buf, int flag);
DIR *opendir(const char *name);
DIR *fdopendir(int fd);
int isatty(int fildes);
typedef struct {
  int x;
} FILE;
FILE *popen(const char *command, const char *type);
int pclose(FILE *stream);
int close(int fildes);
long fpathconf(int fildes, int name);
long pathconf(const char *path, int name);
FILE *fdopen(int fd, const char *mode);
void rewinddir(DIR *dir);
void seekdir(DIR *dirp, long loc);
int rand_r(unsigned int *seedp);
int fileno(FILE *stream);
int fseeko(FILE *stream, off_t offset, int whence);
off_t ftello(FILE *stream);
void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
typedef off_t off64_t;
void *mmap64(void *addr, size_t length, int prot, int flags, int fd, off64_t offset);
int pipe(int fildes[2]);
off_t lseek(int fildes, off_t offset, int whence);
typedef size_t ssize_t;
ssize_t readlink(const char *restrict path, char *restrict buf, size_t bufsize);
ssize_t readlinkat(int fd, const char *restrict path, char *restrict buf, size_t bufsize);
int renameat(int olddirfd, const char *oldpath, int newdirfd, const char *newpath);
char *realpath(const char *restrict file_name, char *restrict resolved_name);
int execv(const char *path, char *const argv[]);
int execvp(const char *file, char *const argv[]);
int getopt(int argc, char *const argv[], const char *optstring);

// In some libc implementations, sockaddr parameter is a transparent
// union of the underlying sockaddr_ pointers instead of being a
// pointer to struct sockaddr.
// We match that with the joker Irrelevant type.
struct sockaddr;
struct sockaddr_at;
#define __SOCKADDR_ALLTYPES    \
  __SOCKADDR_ONETYPE(sockaddr) \
  __SOCKADDR_ONETYPE(sockaddr_at)
#define __SOCKADDR_ONETYPE(type) struct type *__restrict __##type##__;
typedef union {
  __SOCKADDR_ALLTYPES
} __SOCKADDR_ARG __attribute__((__transparent_union__));
#undef __SOCKADDR_ONETYPE
#define __SOCKADDR_ONETYPE(type) const struct type *__restrict __##type##__;
typedef union {
  __SOCKADDR_ALLTYPES
} __CONST_SOCKADDR_ARG __attribute__((__transparent_union__));
#undef __SOCKADDR_ONETYPE
typedef unsigned socklen_t;

int accept(int socket, __SOCKADDR_ARG address, socklen_t *restrict address_len);
int bind(int socket, __CONST_SOCKADDR_ARG address, socklen_t address_len);
int getpeername(int socket, __SOCKADDR_ARG address, socklen_t *restrict address_len);
int getsockname(int socket, __SOCKADDR_ARG address, socklen_t *restrict address_len);
int connect(int socket, __CONST_SOCKADDR_ARG address, socklen_t address_len);
ssize_t recvfrom(int socket, void *restrict buffer, size_t length, int flags, __SOCKADDR_ARG address, socklen_t *restrict address_len);
ssize_t sendto(int socket, const void *message, size_t length, int flags, __CONST_SOCKADDR_ARG dest_addr, socklen_t dest_len);

int listen(int sockfd, int backlog);
ssize_t recv(int sockfd, void *buf, size_t len, int flags);
struct msghdr;
ssize_t recvmsg(int sockfd, struct msghdr *msg, int flags);
ssize_t sendmsg(int sockfd, const struct msghdr *msg, int flags);
int setsockopt(int socket, int level, int option_name, const void *option_value, socklen_t option_len);
int getsockopt(int socket, int level, int option_name, void *restrict option_value, socklen_t *restrict option_len);
ssize_t send(int sockfd, const void *buf, size_t len, int flags);
int socketpair(int domain, int type, int protocol, int sv[2]);
int getnameinfo(const struct sockaddr *restrict sa, socklen_t salen, char *restrict node, socklen_t nodelen, char *restrict service, socklen_t servicelen, int flags);
struct utimbuf;
struct timespec { int x; };
struct timeval { int x; };
int utime(const char *filename, struct utimbuf *buf);
int futimens(int fd, const struct timespec times[2]);
int utimensat(int dirfd, const char *pathname, const struct timespec times[2], int flags);
int utimes(const char *filename, const struct timeval times[2]);
int nanosleep(const struct timespec *rqtp, struct timespec *rmtp);
typedef unsigned long time_t;
struct tm *localtime(const time_t *tp);
struct tm *localtime_r(const time_t *restrict timer, struct tm *restrict result);
char *asctime_r(const struct tm *restrict tm, char *restrict buf);
char *ctime_r(const time_t *timep, char *buf);
struct tm *gmtime_r(const time_t *restrict timer, struct tm *restrict result);
struct tm *gmtime(const time_t *tp);
typedef unsigned long clockid_t;
int clock_gettime(clockid_t clock_id, struct timespec *tp);
struct itimerval;
int getitimer(int which, struct itimerval *curr_value);

typedef union {
  int x;
} pthread_cond_t;
int pthread_cond_signal(pthread_cond_t *cond);
int pthread_cond_broadcast(pthread_cond_t *cond);
typedef union {
  int x;
} pthread_attr_t;
typedef unsigned long int pthread_t;
int pthread_create(pthread_t *restrict thread, const pthread_attr_t *restrict attr, void *(*start_routine)(void *), void *restrict arg);
int pthread_attr_destroy(pthread_attr_t *attr);
int pthread_attr_init(pthread_attr_t *attr);
int pthread_attr_getstacksize(const pthread_attr_t *restrict attr, size_t *restrict stacksize);
int pthread_attr_getguardsize(const pthread_attr_t *restrict attr, size_t *restrict guardsize);
int pthread_attr_setstacksize(pthread_attr_t *attr, size_t stacksize);
int pthread_attr_setguardsize(pthread_attr_t *attr, size_t guardsize);
typedef union {
  int x;
} pthread_mutex_t;
typedef union {
  int x;
} pthread_mutexattr_t;
int pthread_mutex_init(pthread_mutex_t *restrict mutex, const pthread_mutexattr_t *restrict attr);
int pthread_mutex_destroy(pthread_mutex_t *mutex);
int pthread_mutex_lock(pthread_mutex_t *mutex);
int pthread_mutex_trylock(pthread_mutex_t *mutex);
int pthread_mutex_unlock(pthread_mutex_t *mutex);

// Must have at least one call expression to initialize the summary map.
int bar(void);
void foo(void) {
  bar();
}
