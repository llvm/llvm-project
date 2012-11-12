// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
typedef long unsigned int __darwin_size_t;
typedef long __darwin_ssize_t;
typedef __darwin_size_t size_t;
typedef __darwin_ssize_t ssize_t;

struct cmsghdr {};

#if 0
This code below comes from the following system headers:
sys/socket.h:#define CMSG_SPACE(l) (__DARWIN_ALIGN(sizeof(struct  
cmsghdr)) + __DARWIN_ALIGN(l))

i386/_param.h:#define __DARWIN_ALIGN(p) ((__darwin_size_t)((char *)(p)  
+ __DARWIN_ALIGNBYTES) &~ __DARWIN_ALIGNBYTES)
#endif

ssize_t sendFileDescriptor(int fd, void *data, size_t nbytes, int sendfd) {
  union {
    char control[(((__darwin_size_t)((char *)(sizeof(struct cmsghdr)) + (sizeof(__darwin_size_t) - 1)) &~ (sizeof(__darwin_size_t) - 1)) + ((__darwin_size_t)((char *)(sizeof(int)) + (sizeof(__darwin_size_t) - 1)) &~ (sizeof(__darwin_size_t) - 1)))];
  } control_un;
  return 0;
}

