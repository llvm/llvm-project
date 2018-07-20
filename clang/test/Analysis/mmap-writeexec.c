// RUN: %clang_analyze_cc1 -triple i686-unknown-linux -analyzer-checker=alpha.security.MmapWriteExec -analyzer-config alpha.security.MmapWriteExec:MmapProtExec=1 -analyzer-config alpha.security.MmapWriteExec:MmapProtRead=4 -DUSE_ALTERNATIVE_PROT_EXEC_DEFINITION -verify %s
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-apple-darwin10 -analyzer-checker=alpha.security.MmapWriteExec -verify %s

#define PROT_WRITE  0x02
#ifndef USE_ALTERNATIVE_PROT_EXEC_DEFINITION
#define PROT_EXEC   0x04
#define PROT_READ   0x01
#else
#define PROT_EXEC   0x01
#define PROT_READ   0x04
#endif
#define MAP_PRIVATE 0x0002
#define MAP_ANON    0x1000
#define MAP_FIXED   0x0010
#define NULL        ((void *)0)

typedef __typeof(sizeof(int)) size_t;
void *mmap(void *, size_t, int, int, int, long);
int mprotect(void *, size_t, int);

void f1()
{
  void *a = mmap(NULL, 16, PROT_READ | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0); // no-warning
  void *b = mmap(a, 16, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED | MAP_ANON, -1, 0); // no-warning
  void *c = mmap(NULL, 32, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0); // expected-warning{{Both PROT_WRITE and PROT_EXEC flags are set. This can lead to exploitable memory regions, which could be overwritten with malicious code}}
  (void)a;
  (void)b;
  (void)c;
}

void f2()
{
  void *(*callm)(void *, size_t, int, int, int, long);
  callm = mmap;
  int prot = PROT_WRITE | PROT_EXEC;
  (void)callm(NULL, 1024, prot, MAP_PRIVATE | MAP_ANON, -1, 0); // expected-warning{{Both PROT_WRITE and PROT_EXEC flags are set. This can lead to exploitable memory regions, which could be overwritten with malicious code}}
}

void f3()
{
  void *p = mmap(NULL, 1024, PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0); // no-warning
  int m = mprotect(p, 1024, PROT_WRITE | PROT_EXEC); // expected-warning{{Both PROT_WRITE and PROT_EXEC flags are set. This can lead to exploitable memory regions, which could be overwritten with malicious code}}
  (void)m;
}
