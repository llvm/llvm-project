#include "std-c-library-functions.h"

typedef int off_t;
typedef unsigned int mode_t;
typedef __WCHAR_TYPE__ wchar_t;
typedef unsigned int dev_t;
typedef unsigned int uid_t;
typedef unsigned int gid_t;
typedef unsigned long socklen_t;
typedef unsigned long int pthread_t;
typedef unsigned long time_t;
typedef unsigned long clockid_t;
typedef __INT64_TYPE__ off64_t;

typedef struct {
  int a;
} DIR;
struct stat {
  int a;
};
struct timespec { int x; };
struct timeval { int x; };
struct sockaddr;
struct sockaddr_at;
struct msghdr;
struct utimbuf;
struct itimerval;
typedef union {
  int x;
} pthread_cond_t;
typedef union {
  int x;
} pthread_attr_t;
typedef union {
  int x;
} pthread_mutex_t;
typedef union {
  int x;
} pthread_mutexattr_t;

FILE *fopen(const char *restrict pathname, const char *restrict mode);
FILE *tmpfile(void);
FILE *freopen(const char *restrict pathname, const char *restrict mode,
              FILE *restrict stream);
int fclose(FILE *stream);
int fseek(FILE *stream, long offset, int whence);
int fileno(FILE *stream);
long a64l(const char *str64);
char *l64a(long value);
int access(const char *pathname, int amode);
int faccessat(int dirfd, const char *pathname, int mode, int flags);
int dup(int fildes);
int dup2(int fildes1, int filedes2);
int fdatasync(int fildes);
int fnmatch(const char *pattern, const char *string, int flags);
int fsync(int fildes);
int truncate(const char *path, off_t length);
int symlink(const char *oldpath, const char *newpath);
int symlinkat(const char *oldpath, int newdirfd, const char *newpath);
int lockf(int fd, int cmd, off_t len);
int creat(const char *pathname, mode_t mode);
unsigned int sleep(unsigned int seconds);
int dirfd(DIR *dirp);
unsigned int alarm(unsigned int seconds);
int closedir(DIR *dir);
char *strdup(const char *s);
char *strndup(const char *s, size_t n);
wchar_t *wcsdup(const wchar_t *s);
int mkstemp(char *template);
char *mkdtemp(char *template);
char *getcwd(char *buf, size_t size);
int mkdir(const char *pathname, mode_t mode);
int mkdirat(int dirfd, const char *pathname, mode_t mode);
int mknod(const char *pathname, mode_t mode, dev_t dev);
int mknodat(int dirfd, const char *pathname, mode_t mode, dev_t dev);
int chmod(const char *path, mode_t mode);
int fchmodat(int dirfd, const char *pathname, mode_t mode, int flags);
int fchmod(int fildes, mode_t mode);
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
int fstat(int fd, struct stat *statbuf);
int stat(const char *restrict path, struct stat *restrict buf);
int lstat(const char *restrict path, struct stat *restrict buf);
int fstatat(int fd, const char *restrict path, struct stat *restrict buf, int flag);
DIR *opendir(const char *name);
DIR *fdopendir(int fd);
int isatty(int fildes);
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
void *mmap64(void *addr, size_t length, int prot, int flags, int fd, off64_t offset);
int pipe(int fildes[2]);
off_t lseek(int fildes, off_t offset, int whence);
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
#define __SOCKADDR_ALLTYPES    \
  __SOCKADDR_ONETYPE(sockaddr) \
  __SOCKADDR_ONETYPE(sockaddr_at)
#define __SOCKADDR_ONETYPE(type) struct type *restrict __##type##__;
typedef union {
  __SOCKADDR_ALLTYPES
} __SOCKADDR_ARG __attribute__((__transparent_union__));
#undef __SOCKADDR_ONETYPE
#define __SOCKADDR_ONETYPE(type) const struct type *restrict __##type##__;
typedef union {
  __SOCKADDR_ALLTYPES
} __CONST_SOCKADDR_ARG __attribute__((__transparent_union__));
#undef __SOCKADDR_ONETYPE

int accept(int socket, __SOCKADDR_ARG address, socklen_t *restrict address_len);
int bind(int socket, __CONST_SOCKADDR_ARG address, socklen_t address_len);
int getpeername(int socket, __SOCKADDR_ARG address, socklen_t *restrict address_len);
int getsockname(int socket, __SOCKADDR_ARG address, socklen_t *restrict address_len);
int connect(int socket, __CONST_SOCKADDR_ARG address, socklen_t address_len);
ssize_t recvfrom(int socket, void *restrict buffer, size_t length, int flags, __SOCKADDR_ARG address, socklen_t *restrict address_len);
ssize_t sendto(int socket, const void *message, size_t length, int flags, __CONST_SOCKADDR_ARG dest_addr, socklen_t dest_len);
int listen(int sockfd, int backlog);
ssize_t recv(int sockfd, void *buf, size_t len, int flags);
ssize_t recvmsg(int sockfd, struct msghdr *msg, int flags);
ssize_t sendmsg(int sockfd, const struct msghdr *msg, int flags);
int setsockopt(int socket, int level, int option_name, const void *option_value, socklen_t option_len);
int getsockopt(int socket, int level, int option_name, void *restrict option_value, socklen_t *restrict option_len);
ssize_t send(int sockfd, const void *buf, size_t len, int flags);
int socketpair(int domain, int type, int protocol, int sv[2]);
int getnameinfo(const struct sockaddr *restrict sa, socklen_t salen, char *restrict node, socklen_t nodelen, char *restrict service, socklen_t servicelen, int flags);
int utime(const char *filename, struct utimbuf *buf);
int futimens(int fd, const struct timespec times[2]);
int utimensat(int dirfd, const char *pathname, const struct timespec times[2], int flags);
int utimes(const char *filename, const struct timeval times[2]);
int nanosleep(const struct timespec *rqtp, struct timespec *rmtp);
struct tm *localtime(const time_t *tp);
struct tm *localtime_r(const time_t *restrict timer, struct tm *restrict result);
char *asctime_r(const struct tm *restrict tm, char *restrict buf);
char *ctime_r(const time_t *timep, char *buf);
struct tm *gmtime_r(const time_t *restrict timer, struct tm *restrict result);
struct tm *gmtime(const time_t *tp);
int clock_gettime(clockid_t clock_id, struct timespec *tp);
int getitimer(int which, struct itimerval *curr_value);

int pthread_cond_signal(pthread_cond_t *cond);
int pthread_cond_broadcast(pthread_cond_t *cond);
int pthread_create(pthread_t *restrict thread, const pthread_attr_t *restrict attr, void *(*start_routine)(void *), void *restrict arg);
int pthread_attr_destroy(pthread_attr_t *attr);
int pthread_attr_init(pthread_attr_t *attr);
int pthread_attr_getstacksize(const pthread_attr_t *restrict attr, size_t *restrict stacksize);
int pthread_attr_getguardsize(const pthread_attr_t *restrict attr, size_t *restrict guardsize);
int pthread_attr_setstacksize(pthread_attr_t *attr, size_t stacksize);
int pthread_attr_setguardsize(pthread_attr_t *attr, size_t guardsize);
int pthread_mutex_init(pthread_mutex_t *restrict mutex, const pthread_mutexattr_t *restrict attr);
int pthread_mutex_destroy(pthread_mutex_t *mutex);
int pthread_mutex_lock(pthread_mutex_t *mutex);
int pthread_mutex_trylock(pthread_mutex_t *mutex);
int pthread_mutex_unlock(pthread_mutex_t *mutex);
