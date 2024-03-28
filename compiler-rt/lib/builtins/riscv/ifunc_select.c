//=== ifunc_select.c - Check environment hardware feature -*- C -*-===========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define BUFSIZE 1024

static void fillzero(char *s, unsigned size) {
  for (int i = 0; i < size; i++)
    s[i] = '\0';
}

static unsigned strcmp(const char *a, const char *b) {
  while (*a != '\0' && *b != '\0') {
    if (*a != *b)
      return 0;
    a++;
    b++;
  }
  if (*a == *b)
    return 1;
  return 0;
}

static void memcpy(char *dest, const char *src, unsigned len) {
  for (unsigned i = 0; i < len; i++)
    dest[i] = src[i];
}

static unsigned strlen(char *s) {
  int len = 0;
  while (*s != '\0') {
    len++;
    s++;
  }
  return len;
}

// TODO: Replace with more efficient algorithm
static unsigned isPatternInSrc(char *src, char *pattern) {
  char buf[BUFSIZE];
  int lenOfSrc = strlen(src);
  int lenOfPattern = strlen(pattern);
  for (int i = 0; i + lenOfPattern < lenOfSrc; i++) {
    fillzero(buf, BUFSIZE);
    memcpy(buf, src + i, lenOfPattern);
    if (strcmp(buf, pattern))
      return 1;
  }
  return 0;
}

static char *tokenize(char *src, char *rst, char delim) {
  while (*src != '\0' && *src != delim) {
    *rst = *src;
    src++;
    rst++;
  }
  if (*src == delim)
    src++;
  return src;
}

static void getBasicExtFromHwFeatStr(char *rst, char *src) {
  while (*src != 'r') {
    src++;
  }

  while (*src != '_' && *src != '\0') {
    *rst = *src;
    rst++;
    src++;
  }
}

static unsigned checkFeatStrInHwFeatStr(char *FeatStr, char *HwFeatStr) {
  // For basic extension like a,m,f,d...
  if (strlen(FeatStr) == 1) {
    char BasicExt[BUFSIZE];
    fillzero(BasicExt, BUFSIZE);
    getBasicExtFromHwFeatStr(BasicExt, HwFeatStr);
    return isPatternInSrc(BasicExt, FeatStr);
  }
  // For zbb,zihintntl...
  return isPatternInSrc(HwFeatStr, FeatStr);
}

#if defined(__linux__)

#define SYS_read 63
#define SYS_openat 56
#define SYS_riscv_hwprobe 258

static long syscall_impl_3_args(long number, long arg1, long arg2, long arg3) {
  register long a7 __asm__("a7") = number;
  register long a0 __asm__("a0") = arg1;
  register long a1 __asm__("a1") = arg2;
  register long a2 __asm__("a2") = arg3;
  __asm__ __volatile__("ecall\n\t"
                       : "=r"(a0)
                       : "r"(a7), "r"(a0), "r"(a1), "r"(a2)
                       : "memory");
  return a0;
}

static long syscall_impl_5_args(long number, long arg1, long arg2, long arg3,
                                long arg4, long arg5) {
  register long a7 __asm__("a7") = number;
  register long a0 __asm__("a0") = arg1;
  register long a1 __asm__("a1") = arg2;
  register long a2 __asm__("a2") = arg3;
  register long a3 __asm__("a3") = arg4;
  register long a4 __asm__("a4") = arg5;
  __asm__ __volatile__("ecall\n\t"
                       : "=r"(a0)
                       : "r"(a7), "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(a4)
                       : "memory");
  return a0;
}

static unsigned read(int fd, const void *buf, unsigned count) {
  return syscall_impl_3_args(SYS_read, fd, (long)buf, count);
}

static int openat(int dirfd, const char *pathname, int flags) {
  return syscall_impl_3_args(SYS_openat, (long)dirfd, (long)pathname, flags);
}

static void readUntilNextLine(int fd, char *rst, int len) {
  char buf = '\0';
  while (buf != '\n' && read(fd, &buf, 1) != 0 && len > 0) {
    *rst = buf;
    rst++;
    len--;
  }
}

static void __riscv_cpuinfo(char *FeatStr) {
  char buf[BUFSIZE];
  int fd = openat(0, "/proc/cpuinfo", 2);
  do {
    fillzero(buf, BUFSIZE);
    readUntilNextLine(fd, buf, BUFSIZE);
    if (isPatternInSrc(buf, "isa")) {
      memcpy(FeatStr, buf, BUFSIZE);
      return;
    }
  } while (strlen(buf) != 0);
}

struct riscv_hwprobe {
  long long key;
  unsigned long long value;
};

#define RISCV_HWPROBE_MAX_KEY 5
#define RISCV_HWPROBE_KEY_MVENDORID 0
#define RISCV_HWPROBE_KEY_MARCHID 1
#define RISCV_HWPROBE_KEY_MIMPID 2
#define RISCV_HWPROBE_KEY_BASE_BEHAVIOR 3
#define RISCV_HWPROBE_BASE_BEHAVIOR_IMA (1 << 0)
#define RISCV_HWPROBE_KEY_IMA_EXT_0 4
#define RISCV_HWPROBE_IMA_FD (1 << 0)
#define RISCV_HWPROBE_IMA_C (1 << 1)
#define RISCV_HWPROBE_IMA_V (1 << 2)
#define RISCV_HWPROBE_EXT_ZBA (1 << 3)
#define RISCV_HWPROBE_EXT_ZBB (1 << 4)
#define RISCV_HWPROBE_EXT_ZBS (1 << 5)
#define RISCV_HWPROBE_KEY_CPUPERF_0 5
#define RISCV_HWPROBE_MISALIGNED_UNKNOWN (0 << 0)
#define RISCV_HWPROBE_MISALIGNED_EMULATED (1 << 0)
#define RISCV_HWPROBE_MISALIGNED_SLOW (2 << 0)
#define RISCV_HWPROBE_MISALIGNED_FAST (3 << 0)
#define RISCV_HWPROBE_MISALIGNED_UNSUPPORTED (4 << 0)
#define RISCV_HWPROBE_MISALIGNED_MASK (7 << 0)

/* Size definition for CPU sets.  */
#define __CPU_SETSIZE 1024
#define __NCPUBITS (8 * sizeof(unsigned long int))

/* Data structure to describe CPU mask.  */
typedef struct {
  unsigned long int __bits[__CPU_SETSIZE / __NCPUBITS];
} cpu_set_t;

static long sys_riscv_hwprobe(struct riscv_hwprobe *pairs, unsigned pair_count,
                              unsigned cpu_count, cpu_set_t *cpus,
                              unsigned int flags) {
  return syscall_impl_5_args(SYS_riscv_hwprobe, (long)pairs, pair_count,
                             cpu_count, (long)cpus, flags);
}

static void initHwProbe(struct riscv_hwprobe *Hwprobes, int len) {
  sys_riscv_hwprobe(Hwprobes, len, 0, 0, 0);
}

static unsigned checkFeatStrInHwProbe(char *FeatStr,
                                      struct riscv_hwprobe Hwprobe) {
  if (Hwprobe.key == -1)
    return 0;

  if (strcmp(FeatStr, "i"))
    return Hwprobe.value & RISCV_HWPROBE_KEY_IMA_EXT_0;
  if (strcmp(FeatStr, "m"))
    return Hwprobe.value & RISCV_HWPROBE_KEY_IMA_EXT_0;
  if (strcmp(FeatStr, "a"))
    return Hwprobe.value & RISCV_HWPROBE_KEY_IMA_EXT_0;
  if (strcmp(FeatStr, "f"))
    return Hwprobe.value & RISCV_HWPROBE_IMA_FD;
  if (strcmp(FeatStr, "d"))
    return Hwprobe.value & RISCV_HWPROBE_IMA_FD;
  if (strcmp(FeatStr, "c"))
    return Hwprobe.value & RISCV_HWPROBE_IMA_C;
  if (strcmp(FeatStr, "v"))
    return Hwprobe.value & RISCV_HWPROBE_IMA_V;
  if (strcmp(FeatStr, "zba"))
    return Hwprobe.value & RISCV_HWPROBE_EXT_ZBA;
  if (strcmp(FeatStr, "zbb"))
    return Hwprobe.value & RISCV_HWPROBE_EXT_ZBB;
  if (strcmp(FeatStr, "zbs"))
    return Hwprobe.value & RISCV_HWPROBE_EXT_ZBS;

  return 0;
}
#endif // defined(__linux__)

// FeatStr format is like <Feature1>;<Feature2>...<FeatureN>.
unsigned __riscv_ifunc_select(char *FeatStrs) {
#if defined(__linux__)
  // Init Hwprobe
  struct riscv_hwprobe pairs[] = {
      {RISCV_HWPROBE_KEY_IMA_EXT_0, 0},
  };
  initHwProbe(pairs, 1);
  // Init from cpuinfo
  char HwFeatStr[BUFSIZE];
  fillzero(HwFeatStr, BUFSIZE);
  __riscv_cpuinfo(HwFeatStr);

  // Check each extension whether available
  char FeatStr[BUFSIZE];
  while (*FeatStrs != '\0') {
    fillzero(FeatStr, BUFSIZE);
    FeatStrs = tokenize(FeatStrs, FeatStr, ';');
    if (checkFeatStrInHwProbe(FeatStr, pairs[0]))
      continue;
    if (!checkFeatStrInHwFeatStr(FeatStr, HwFeatStr))
      return 0;
  }

  return 1;
#else
  // If other platform support IFUNC, need to implement its
  // __riscv_ifunc_select.
  return 0;
#endif
}
