// Check that shadow of retrieved value from va_list matches the shadow of passed value.

// Without -fno-sanitize-memory-param-retval we can't even pass poisoned values.
// RUN: %clangxx_msan -fno-sanitize-memory-param-retval -fsanitize-memory-track-origins=0 -O3 %s -o %t

// FIXME: The rest is likely still broken.
// XFAIL: target={{(loongarch64|mips|powerpc64).*}}

#include <sanitizer/msan_interface.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef DEBUG_VARARG_SHADOW_TEST
__attribute__((noinline, no_sanitize("memory"))) void
printb(const void *p, size_t n, int line, int align) {
  fprintf(stderr, "\n%p at line %d: \n", p, line);
  for (int i = 0; i < n;) {
    fprintf(stderr, "%p: ", (void *)(((uint8_t *)p) + i));
    for (int j = 0; j < align; ++i, ++j)
      fprintf(stderr, "%02x ", ((uint8_t *)p)[i]);
    fprintf(stderr, "\n");
  }
}

struct my_va_list {
#  ifdef __ARM_ARCH_ISA_A64
  void *stack;
  void *gr_top;
  void *vr_top;
  int gr_offs;
  int vr_offs;
#  else
  unsigned int gp_offset;
  unsigned int fp_offset;
  void *overflow_arg_area;
  void *reg_save_area;
#  endif
};

__attribute__((noinline, no_sanitize("memory"))) void printva(const void *p,
                                                              int line) {
  my_va_list *pp = (my_va_list *)p;
#  ifdef __ARM_ARCH_ISA_A64
  fprintf(stderr,
          "\nva %p at line %d: stack : %p\n gr_top: %p\n vr_top: %p\n gr_offs: "
          "%d\n "
          "vr_offs: %d\n",
          p, line, pp->stack, pp->gr_top, pp->vr_top, pp->gr_offs, pp->vr_offs);

  printb((char *)pp->gr_top + pp->gr_offs, -pp->gr_offs, __LINE__, 8);
  printb((char *)pp->vr_top + pp->vr_offs, -pp->vr_offs, __LINE__, 16);
  printb((char *)pp->stack, 256, __LINE__, 8);
#  else
  fprintf(stderr,
          "\nva %p at line %d:\n gp_offset: %u\n fp_offset: %u\n "
          "overflow_arg_area: %p\n reg_save_area: %p\n\n",
          p, line, pp->gp_offset, pp->fp_offset, pp->overflow_arg_area,
          pp->reg_save_area);

  printb((char *)pp->reg_save_area + pp->gp_offset,
         pp->fp_offset - pp->gp_offset, __LINE__, 8);
  printb((char *)pp->reg_save_area + pp->fp_offset, 128, __LINE__, 16);
  printb((char *)pp->overflow_arg_area, 256, __LINE__, 8);
#  endif
}

__attribute__((noinline, no_sanitize("memory"))) void printtls(int line) {
  uint8_t tmp[kMsanParamTlsSize];
  for (int i = 0; i < kMsanParamTlsSize; ++i)
    tmp[i] = __msan_va_arg_tls[i];
  fprintf(stderr, "\nTLS at line %d: ", line);
  for (int i = 0; i < kMsanParamTlsSize;) {
    fprintf(stderr, "\n");
    for (int j = 0; j < 16; ++i, ++j)
      fprintf(stderr, "%02x ", tmp[i]);
  }

  fprintf(stderr, "\n");
}
#endif // DEBUG_VARARG_SHADOW_TEST

const int kMsanParamTlsSize = 800;
extern "C" __thread uint8_t __msan_va_arg_tls[];

struct IntInt {
  int a;
  int b;
};

struct Int64Int64 {
  int64_t a;
  int64_t b;
};

struct DoubleDouble {
  double a;
  double b;
};

struct Double4 {
  double a[4];
};

struct DoubleFloat {
  double a;
  float b;
};

struct LongDouble2 {
  long double a[2];
};

struct LongDouble4 {
  long double a[4];
};

template <class T>
__attribute__((noinline)) void print_shadow(va_list &args, int n,
                                            const char *function) {
  for (int i = 0; i < n; i++) {
    // 1-based to make it different from clean shadow.
    fprintf(stderr, "\nArgShadow fn:%s n:%d i:%02x ", function, n, i + 1);
    T arg_int = va_arg(args, T);
    if (__msan_test_shadow(&arg_int, sizeof(arg_int)))
      fprintf(stderr, "fake[clean] %02x", i + 1);
    else
      __msan_dump_shadow(&arg_int, sizeof(arg_int));
#ifdef DEBUG_VARARG_SHADOW_TEST
    printb(&arg_int, sizeof(arg_int), __LINE__, 16);
#endif
  }
}

template <class T> __attribute__((noinline)) void test1(int n, ...) {
#ifdef DEBUG_VARARG_SHADOW_TEST
  printtls(__LINE__);
#endif
  va_list args;
  va_start(args, n);
#ifdef DEBUG_VARARG_SHADOW_TEST
  printva(&args, __LINE__);
#endif
  print_shadow<T>(args, n, __FUNCTION__);
  va_end(args);
}

template <class T> __attribute__((noinline)) void test2(T t, int n, ...) {
#ifdef DEBUG_VARARG_SHADOW_TEST
  printtls(__LINE__);
#endif
  va_list args;
  va_start(args, n);
#ifdef DEBUG_VARARG_SHADOW_TEST
  printva(&args, __LINE__);
#endif
  print_shadow<T>(args, n, __FUNCTION__);
  va_end(args);
}

template <class T> __attribute__((noinline)) void test() {
  // Array of values we will pass into variadic functions.
  static T args[32] = {};

  // Poison values making the fist byte of the item shadow match the index.
  // E.g. item 3 should be poisoned as '03 ff ff ff'.
  memset(args, 0xff, sizeof(args));
  __msan_poison(args, sizeof(args));
  for (int i = 0; i < 32; ++i) {
    char *first = (char *)(&args[i]);
    *first = char(*(int *)(first)&i);
  }
#ifdef DEBUG_VARARG_SHADOW_TEST
  __msan_print_shadow(args, sizeof(args));
#endif

  // Now we will check that index, printed like 'i:03' will match
  // '0x123abc[0x123abc] 03 ff ff ff'
  memset(__msan_va_arg_tls, 0xee, kMsanParamTlsSize);
  test1<T>(1, args[1]);
  // CHECK-COUNT-1: ArgShadow fn:test1 n:1 i:[[ARGI:[[:xdigit:]]{2}]] {{[^]]+}}] [[ARGI]]

  memset(__msan_va_arg_tls, 0xee, kMsanParamTlsSize);
  test1<T>(4, args[1], args[2], args[3], args[4]);
  // CHECK-COUNT-4: ArgShadow fn:test1 n:4 i:[[ARGI:[[:xdigit:]]{2}]] {{[^]]+}}] [[ARGI]]

  memset(__msan_va_arg_tls, 0xee, kMsanParamTlsSize);
  test1<T>(20, args[1], args[2], args[3], args[4], args[5], args[6], args[7],
           args[8], args[9], args[10], args[11], args[12], args[13], args[14],
           args[15], args[16], args[17], args[18], args[19], args[20]);
  // CHECK-COUNT-20: ArgShadow fn:test1 n:20 i:[[ARGI:[[:xdigit:]]{2}]] {{[^]]+}}] [[ARGI]]

  memset(__msan_va_arg_tls, 0xee, kMsanParamTlsSize);
  test2<T>(args[31], 1, args[1]);
  // CHECK-COUNT-1: ArgShadow fn:test2 n:1 i:[[ARGI:[[:xdigit:]]{2}]] {{[^]]+}}] [[ARGI]]

  memset(__msan_va_arg_tls, 0xee, kMsanParamTlsSize);
  test2<T>(args[31], 4, args[1], args[2], args[3], args[4]);
  // CHECK-COUNT-4: ArgShadow fn:test2 n:4 i:[[ARGI:[[:xdigit:]]{2}]] {{[^]]+}}] [[ARGI]]

  memset(__msan_va_arg_tls, 0xee, kMsanParamTlsSize);
  test2<T>(args[31], 20, args[1], args[2], args[3], args[4], args[5], args[6],
           args[7], args[8], args[9], args[10], args[11], args[12], args[13],
           args[14], args[15], args[16], args[17], args[18], args[19],
           args[20]);
  // CHECK-COUNT-20: ArgShadow fn:test2 n:20 i:[[ARGI:[[:xdigit:]]{2}]] {{[^]]+}}] [[ARGI]]
}

int main(int argc, char *argv[]) {
#define TEST(T...)                                                             \
  if (argc == 2 && strcmp(argv[1], #T) == 0) {                                 \
    test<T>();                                                                 \
    return 0;                                                                  \
  }

  TEST(char);
  // RUN: %run %t char 2>&1 | FileCheck %s --implicit-check-not="ArgShadow"

  TEST(int);
  // RUN: %run %t int 2>&1 | FileCheck %s --implicit-check-not="ArgShadow"

  TEST(void*);
  // RUN: %run %t "void*" 2>&1 | FileCheck %s --implicit-check-not="ArgShadow"

  TEST(float);
  // RUN: %run %t float 2>&1 | FileCheck %s --implicit-check-not="ArgShadow"

  TEST(double);
  // RUN: %run %t double 2>&1 | FileCheck %s --implicit-check-not="ArgShadow"

  TEST(long double);
  // RUN: %run %t "long double" 2>&1 | FileCheck %s --implicit-check-not="ArgShadow"

  TEST(IntInt);
  // RUN: %run %t IntInt 2>&1 | FileCheck %s --implicit-check-not="ArgShadow"

  TEST(Int64Int64);
  // RUN: %run %t Int64Int64 2>&1 | FileCheck %s --implicit-check-not="ArgShadow"

  TEST(DoubleDouble);
  // RUN: %run %t DoubleDouble 2>&1 | FileCheck %s --implicit-check-not="ArgShadow"

  TEST(Double4);
  // RUN: %run %t Double4 2>&1 | FileCheck %s --implicit-check-not="ArgShadow"

  TEST(DoubleFloat);
  // RUN: %run %t DoubleFloat 2>&1 | FileCheck %s --implicit-check-not="ArgShadow"

  TEST(LongDouble2);
  // RUN: %run %t LongDouble2 2>&1 | FileCheck %s --implicit-check-not="ArgShadow"

  TEST(LongDouble4);
  // RUN: %run %t LongDouble4 2>&1 | FileCheck %s --implicit-check-not="ArgShadow"

  return 1;
}
