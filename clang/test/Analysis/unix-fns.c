// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,unix.API,osx.API,optin.portability %s -analyzer-output=plist -analyzer-config faux-bodies=true  -fblocks -verify -o %t.plist
// RUN: %normalize_plist <%t.plist | diff -ub %S/Inputs/expected-plists/unix-fns.c.plist -
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux -analyzer-checker=core,unix.API,osx.API,optin.portability %s -analyzer-output=plist -analyzer-config faux-bodies=true  -fblocks -verify -o %t.plist
// RUN: %normalize_plist <%t.plist | diff -ub %S/Inputs/expected-plists/unix-fns.c.plist -
// RUN: mkdir -p %t.dir
// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.API,osx.API,optin.portability -analyzer-output=html -analyzer-config faux-bodies=true -fblocks -o %t.dir %s
// RUN: rm -fR %t.dir


struct _opaque_pthread_once_t {
  long __sig;
  char __opaque[8];
};
typedef struct _opaque_pthread_once_t    __darwin_pthread_once_t;
typedef __darwin_pthread_once_t pthread_once_t;
int pthread_once(pthread_once_t *, void (*)(void));
typedef long unsigned int __darwin_size_t;
typedef __darwin_size_t size_t;
void *calloc(size_t, size_t);
void *malloc(size_t);
void *realloc(void *, size_t);
void *reallocf(void *, size_t);
void *alloca(size_t);
void *valloc(size_t);
typedef union {
 struct _os_object_s *_os_obj;
 struct dispatch_object_s *_do;
 struct dispatch_continuation_s *_dc;
 struct dispatch_queue_s *_dq;
 struct dispatch_queue_attr_s *_dqa;
 struct dispatch_group_s *_dg;
 struct dispatch_source_s *_ds;
 struct dispatch_source_attr_s *_dsa;
 struct dispatch_semaphore_s *_dsema;
 struct dispatch_data_s *_ddata;
 struct dispatch_io_s *_dchannel;
 struct dispatch_operation_s *_doperation;
 struct dispatch_disk_s *_ddisk;
} dispatch_object_t __attribute__((__transparent_union__));
typedef void (^dispatch_block_t)(void);
typedef struct dispatch_queue_s *dispatch_queue_t;
void dispatch_sync(dispatch_queue_t queue, dispatch_block_t block);

typedef long dispatch_once_t;

extern
__attribute__((__nonnull__))
__attribute__((__nothrow__))
void dispatch_once(dispatch_once_t *predicate,
                   __attribute__((__noescape__)) dispatch_block_t block);

// Inlined fast-path dispatch_once defers to the real dispatch_once
// on the slow path.
static
__inline__
__attribute__((__always_inline__))
__attribute__((__nonnull__))
__attribute__((__nothrow__))
void _dispatch_once(dispatch_once_t *predicate,
                    __attribute__((__noescape__)) dispatch_block_t block)
{
 if (__builtin_expect((*predicate), (~0l)) != ~0l) {
  dispatch_once(predicate, block);
 } else {
  __asm__ __volatile__("" ::: "memory");
 }
 __builtin_assume(*predicate == ~0l);
}

// Macro so that user calls to dispatch_once() call the inlined fast-path
// variant.
#undef dispatch_once
#define dispatch_once _dispatch_once

#ifndef O_CREAT
#define O_CREAT 0x0200
#define O_RDONLY 0x0000
#endif
int open(const char *, int, ...);
int openat(int, const char *, int, ...);
int close(int fildes);

void test_open(const char *path) {
  int fd;
  fd = open(path, O_RDONLY); // no-warning
  if (!fd)
    close(fd);

  fd = open(path, O_CREAT); // expected-warning{{Call to 'open' requires a 3rd argument when the 'O_CREAT' flag is set}}
  if (!fd)
    close(fd);
} 

void test_open_at(int directory_fd, const char *relative_path) {
  int fd;
  fd = openat(directory_fd, relative_path, O_RDONLY); // no-warning
  if (!fd)
    close(fd);

  fd = openat(directory_fd, relative_path, O_CREAT); // expected-warning{{Call to 'openat' requires a 4th argument when the 'O_CREAT' flag is set}}
  if (!fd)
    close(fd);
}

void test_dispatch_once(void) {
  dispatch_once_t pred = 0;
  do { if (__builtin_expect(*(&pred), ~0l) != ~0l) dispatch_once((&pred), (^(void) {})); } while (0); // expected-warning{{Call to 'dispatch_once' uses the local variable 'pred' for the predicate value}}
}
void test_dispatch_once_neg(void) {
  static dispatch_once_t pred = 0;
  do { if (__builtin_expect(*(&pred), ~0l) != ~0l) dispatch_once((&pred), (^(void) {})); } while (0); // no-warning
}

void test_pthread_once_aux(void);

void test_pthread_once(void) {
  pthread_once_t pred = {0x30B1BCBA, {0}};
  pthread_once(&pred, test_pthread_once_aux); // expected-warning{{Call to 'pthread_once' uses the local variable 'pred' for the "control" value}}
}
void test_pthread_once_neg(void) {
  static pthread_once_t pred = {0x30B1BCBA, {0}};
  pthread_once(&pred, test_pthread_once_aux); // no-warning
}

// PR 2899 - warn of zero-sized allocations to malloc().
void pr2899(void) {
  char* foo = malloc(0); // expected-warning{{Call to 'malloc' has an allocation size of 0 bytes}}
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void pr2899_nowarn(size_t size) {
  char* foo = malloc(size); // no-warning
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_calloc(void) {
  char *foo = calloc(0, 42); // expected-warning{{Call to 'calloc' has an allocation size of 0 bytes}}
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_calloc2(void) {
  char *foo = calloc(42, 0); // expected-warning{{Call to 'calloc' has an allocation size of 0 bytes}}
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_calloc_nowarn(size_t nmemb, size_t size) {
  char *foo = calloc(nmemb, size); // no-warning
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_realloc(char *ptr) {
  char *foo = realloc(ptr, 0); // expected-warning{{Call to 'realloc' has an allocation size of 0 bytes}}
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_reallocf(char *ptr) {
  char *foo = reallocf(ptr, 0); // expected-warning{{Call to 'reallocf' has an allocation size of 0 bytes}}
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_realloc_nowarn(char *ptr, size_t size) {
  char *foo = realloc(ptr, size); // no-warning
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_reallocf_nowarn(char *ptr, size_t size) {
  char *foo = reallocf(ptr, size); // no-warning
  for (unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_alloca(void) {
  char *foo = alloca(0); // expected-warning{{Call to 'alloca' has an allocation size of 0 bytes}}
  for(unsigned i = 0; i < 100; i++) {
    foo[i] = 0; 
  }
}
void test_alloca_nowarn(size_t sz) {
  char *foo = alloca(sz); // no-warning
  for(unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}
void test_builtin_alloca(void) {
  char *foo2 = __builtin_alloca(0); // expected-warning{{Call to 'alloca' has an allocation size of 0 bytes}}
  for(unsigned i = 0; i < 100; i++) {
    foo2[i] = 0; 
  }
}
void test_builtin_alloca_nowarn(size_t sz) {
  char *foo2 = __builtin_alloca(sz); // no-warning
  for(unsigned i = 0; i < 100; i++) {
    foo2[i] = 0;
  }
}
void test_valloc(void) {
  char *foo = valloc(0); // expected-warning{{Call to 'valloc' has an allocation size of 0 bytes}}
  for(unsigned i = 0; i < 100; i++) {
    foo[i] = 0; 
  }
}
void test_valloc_nowarn(size_t sz) {
  char *foo = valloc(sz); // no-warning
  for(unsigned i = 0; i < 100; i++) {
    foo[i] = 0;
  }
}

void test_dispatch_once_in_macro(void) {
  dispatch_once_t pred = 0;
  dispatch_once(&pred, ^(void){});  // expected-warning {{Call to 'dispatch_once' uses the local variable 'pred' for the predicate value}}
}

// Test inlining of dispatch_sync.
void test_dispatch_sync(dispatch_queue_t queue, int *q) {
  int *p = 0;
  dispatch_sync(queue, ^(void){ 
	  if (q) {
		*p = 1; // expected-warning {{null pointer}}
	   }
  });
}

// Test inlining of dispatch_once.
void test_inline_dispatch_once(void) {
  static dispatch_once_t pred;
  int *p = 0;
  dispatch_once(&pred, ^(void) {
	  *p = 1; // expected-warning {{null}}
  });
}

// Make sure code after call to dispatch once is reached.
void test_inline_dispatch_once_reachable(void) {
  static dispatch_once_t pred;
  __block int *p;
  dispatch_once(&pred, ^(void) {
      p = 0;
  });

  *p = 7; // expected-warning {{Dereference of null pointer (loaded from variable 'p')}}
}
