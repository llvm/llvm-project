// RUN: %clang_analyze_cc1 -analyzer-checker=security.MmapWriteExec -verify %s

#define PROT_READ   0x01
#define PROT_WRITE  0x02
#define PROT_EXEC   0x04
#define MAP_PRIVATE 0x0002
#define MAP_ANON    0x1000
#define MAP_FIXED   0x0010
#define NULL        ((void *)0)

typedef __typeof(sizeof(int)) size_t;
void *mmap(void *, size_t, int, int, int, long);

void f1()
{
  void *a = mmap(NULL, 16, PROT_READ | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0); // no-warning
  void *b = mmap(a, 16, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_FIXED | MAP_ANON, -1, 0); // no-warning
  void *c = mmap(NULL, 32, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0); // expected-warning{{Both PROT_WRITE and PROT_EXEC flags had been set. It can lead to exploitable memory regions, overwritten with malicious code}}
}

void f2()
{
  void *(*callm)(void *, size_t, int, int, int, long);
  callm = mmap;
  int prot = PROT_WRITE | PROT_EXEC;
  (void)callm(NULL, 1024, prot, MAP_PRIVATE | MAP_ANON, -1, 0); // expected-warning{{Both PROT_WRITE and PROT_EXEC flags had been set. It can lead to exploitable memory regions, overwritten with malicious code}}
}
