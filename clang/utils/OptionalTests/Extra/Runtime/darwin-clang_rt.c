/* This file tests that we can successfully call each compiler-rt function. It is
   designed to check that the runtime libraries are available for linking and
   that they contain the expected contents. It is not designed to test the
   correctness of the individual functions in compiler-rt.

   This test is assumed to be run on a 10.6 machine. The two environment
   variables below should be set to 10.4 and 10.5 machines which can be directly
   ssh/rsync'd to in order to actually test the executables can run on the
   desired targets.
*/

// RUN: export TENFOUR_X86_MACHINE=localhost
// RUN: export TENFIVE_X86_MACHINE=localhost
// RUN: export ARM_MACHINE=localhost
// RUN: export ARM_SYSROOT=$(xcodebuild -sdk iphoneos -version Path)

// RUN: echo iPhoneOS, ARM, v6, thumb
// RUN: %clang -isysroot $ARM_SYSROOT -arch armv6 -mthumb -c %s -o %t.o
// RUN: %clang -isysroot $ARM_SYSROOT -arch armv6 -mthumb -v -Wl,-t,-v -o %t %t.o 1>&2
// RUN: rsync -arv %t $ARM_MACHINE:/tmp/a.out
// RUN: ssh $ARM_MACHINE /tmp/a.out
// RUN: echo

// RUN: echo iPhoneOS, ARM, v6, no-thumb
// RUN: %clang -isysroot $ARM_SYSROOT -arch armv6 -mno-thumb -c %s -o %t.o
// RUN: %clang -isysroot $ARM_SYSROOT -arch armv6 -mno-thumb -v -Wl,-t,-v -o %t %t.o 1>&2
// RUN: rsync -arv %t $ARM_MACHINE:/tmp/a.out
// RUN: ssh $ARM_MACHINE /tmp/a.out
// RUN: echo

// RUN: echo iPhoneOS, ARM, v7, thumb
// RUN: %clang -isysroot $ARM_SYSROOT -arch armv7 -mthumb -c %s -o %t.o
// RUN: %clang -isysroot $ARM_SYSROOT -arch armv7 -mthumb -v -Wl,-t,-v -o %t %t.o 1>&2
// RUN: rsync -arv %t $ARM_MACHINE:/tmp/a.out
// RUN: ssh $ARM_MACHINE /tmp/a.out
// RUN: echo

// RUN: echo iPhoneOS, ARM, v7, no-thumb
// RUN: %clang -isysroot $ARM_SYSROOT -arch armv7 -mno-thumb -c %s -o %t.o
// RUN: %clang -isysroot $ARM_SYSROOT -arch armv7 -mno-thumb -v -Wl,-t,-v -o %t %t.o 1>&2
// RUN: rsync -arv %t $ARM_MACHINE:/tmp/a.out
// RUN: ssh $ARM_MACHINE /tmp/a.out
// RUN: echo

// RUN: echo 10.4, i386
// RUN: %clang -arch i386 -mmacosx-version-min=10.4 -c %s -o %t.o
// RUN: %clang -arch i386 -mmacosx-version-min=10.4 -v -Wl,-t,-v -o %t %t.o 1>&2
// RUN: %t
// RUN: echo

// RUN: rsync -arv %t $TENFOUR_X86_MACHINE:/tmp/a.out
// RUN: ssh $TENFOUR_X86_MACHINE /tmp/a.out
// RUN: echo

// RUX: rsync -arv %t $TENFIVE_X86_MACHINE:/tmp/a.out
// RUX: ssh $TENFIVE_X86_MACHINE /tmp/a.out
// RUN: echo

// RUN: echo 10.5, i386
// RUN: %clang -arch i386 -mmacosx-version-min=10.5 -c %s -o %t.o
// RUN: %clang -arch i386 -mmacosx-version-min=10.5 -v -Wl,-t,-v -o %t %t.o 1>&2
// RUN: %t
// RUN: echo

// RUN: rsync -arv %t $TENFIVE_X86_MACHINE:/tmp/a.out
// RUN: ssh $TENFIVE_X86_MACHINE /tmp/a.out
// RUN: echo

// RUN: echo 10.6, i386
// RUN: %clang -arch i386 -mmacosx-version-min=10.6 -c %s -o %t.o
// RUN: %clang -arch i386 -mmacosx-version-min=10.6 -v -Wl,-t,-v -o %t %t.o 1>&2
// RUN: %t
// RUN: echo

// RUN: echo 10.4, x86_64
// RUN: %clang -arch x86_64 -mmacosx-version-min=10.4 -c %s -o %t.o
// RUN: %clang -arch x86_64 -mmacosx-version-min=10.4 -v -Wl,-t,-v -o %t %t.o 1>&2
// RUN: %t
// RUN: echo

// RUN: rsync -arv %t $TENFOUR_X86_MACHINE:/tmp/a.out
// RUN: ssh $TENFOUR_X86_MACHINE /tmp/a.out
// RUN: echo

// RUN: rsync -arv %t $TENFIVE_X86_MACHINE:/tmp/a.out
// RUN: ssh $TENFIVE_X86_MACHINE /tmp/a.out
// RUN: echo

// RUN: echo 10.5, x86_64
// RUN: %clang -arch x86_64 -mmacosx-version-min=10.5 -c %s -o %t.o
// RUN: %clang -arch x86_64 -mmacosx-version-min=10.5 -v -Wl,-t,-v -o %t %t.o 1>&2
// RUN: %t
// RUN: echo

// RUN: rsync -arv %t $TENFIVE_X86_MACHINE:/tmp/a.out
// RUN: ssh $TENFIVE_X86_MACHINE /tmp/a.out
// RUN: echo

// RUN: echo 10.6, x86_64
// RUN: %clang -arch x86_64 -mmacosx-version-min=10.6 -c %s -o %t.o
// RUN: %clang -arch x86_64 -mmacosx-version-min=10.6 -v -Wl,-t,-v -o %t %t.o 1>&2
// RUN: %t
// RUN: echo

#include <assert.h>
#include <stdio.h>
#include <sys/utsname.h>

typedef int si_int;
typedef unsigned su_int;

typedef long long di_int;
typedef unsigned long long du_int;

// Integral bit manipulation

di_int __ashldi3(di_int a, si_int b);      // a << b
di_int __ashrdi3(di_int a, si_int b);      // a >> b  arithmetic (sign fill)
di_int __lshrdi3(di_int a, si_int b);      // a >> b  logical    (zero fill)

si_int __clzsi2(si_int a);  // count leading zeros
si_int __clzdi2(di_int a);  // count leading zeros
si_int __ctzsi2(si_int a);  // count trailing zeros
si_int __ctzdi2(di_int a);  // count trailing zeros

si_int __ffsdi2(di_int a);  // find least significant 1 bit

si_int __paritysi2(si_int a);  // bit parity
si_int __paritydi2(di_int a);  // bit parity

si_int __popcountsi2(si_int a);  // bit population
si_int __popcountdi2(di_int a);  // bit population

// Integral arithmetic

di_int __negdi2    (di_int a);                         // -a
di_int __muldi3    (di_int a, di_int b);               // a * b
di_int __divdi3    (di_int a, di_int b);               // a / b   signed
du_int __udivdi3   (du_int a, du_int b);               // a / b   unsigned
di_int __moddi3    (di_int a, di_int b);               // a % b   signed
du_int __umoddi3   (du_int a, du_int b);               // a % b   unsigned
du_int __udivmoddi4(du_int a, du_int b, du_int* rem);  // a / b, *rem = a % b

//  Integral arithmetic with trapping overflow

si_int __absvsi2(si_int a);           // abs(a)
di_int __absvdi2(di_int a);           // abs(a)

si_int __negvsi2(si_int a);           // -a
di_int __negvdi2(di_int a);           // -a

si_int __addvsi3(si_int a, si_int b);  // a + b
di_int __addvdi3(di_int a, di_int b);  // a + b

si_int __subvsi3(si_int a, si_int b);  // a - b
di_int __subvdi3(di_int a, di_int b);  // a - b

si_int __mulvsi3(si_int a, si_int b);  // a * b
di_int __mulvdi3(di_int a, di_int b);  // a * b

//  Integral comparison: a  < b -> 0
//                       a == b -> 1
//                       a  > b -> 2

si_int __cmpdi2 (di_int a, di_int b);
si_int __ucmpdi2(du_int a, du_int b);

//  Integral / floating point conversion

di_int __fixsfdi(      float a);
di_int __fixdfdi(     double a);
di_int __fixxfdi(long double a);

su_int __fixunssfsi(      float a);
su_int __fixunsdfsi(     double a);
su_int __fixunsxfsi(long double a);

du_int __fixunssfdi(      float a);
du_int __fixunsdfdi(     double a);
du_int __fixunsxfdi(long double a);

float       __floatdisf(di_int a);
double      __floatdidf(di_int a);
long double __floatdixf(di_int a);

float       __floatundisf(du_int a);
double      __floatundidf(du_int a);
long double __floatundixf(du_int a);

//  Floating point raised to integer power

float       __powisf2(      float a, si_int b);  // a ^ b
double      __powidf2(     double a, si_int b);  // a ^ b
long double __powixf2(long double a, si_int b);  // a ^ b

//  Complex arithmetic

//  (a + ib) * (c + id)

      float _Complex __mulsc3( float a,  float b,  float c,  float d);
     double _Complex __muldc3(double a, double b, double c, double d);
long double _Complex __mulxc3(long double a, long double b,
                              long double c, long double d);

//  (a + ib) / (c + id)

      float _Complex __divsc3( float a,  float b,  float c,  float d);
     double _Complex __divdc3(double a, double b, double c, double d);
long double _Complex __divxc3(long double a, long double b,
                              long double c, long double d);

#ifndef __arm
#define HAS_LONG_DOUBLE
#endif

int main(int argc, char **argv) {
  du_int du_tmp;
  struct utsname name;
#ifdef __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__
  const char *target_name = "OS X";
  unsigned target_version = __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__;
  unsigned target_maj = target_version / 100;
  unsigned target_min = (target_version / 10) % 10;
  unsigned target_micro = target_version % 10;
#else
  const char *target_name = "iPhoneOS";
  unsigned target_version = __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__;
  unsigned target_maj = target_version / 10000;
  unsigned target_min = (target_version / 100) % 100;
  unsigned target_micro = target_version % 100;
#endif

  if (uname(&name))
    return 1;

  fprintf(stderr, "%s: clang_rt test:\n", argv[0]);
  fprintf(stderr, "  target  : %s %d.%d.%d\n\n", target_name,
          target_maj, target_min, target_micro);
  fprintf(stderr, "  sysname : %s\n", name.sysname);
  fprintf(stderr, "  nodename: %s\n", name.nodename);
  fprintf(stderr, "  release : %s\n", name.release);
  fprintf(stderr, "  version : %s\n", name.version);
  fprintf(stderr, "  machine : %s\n", name.machine);

  assert(__ashldi3(1, 1) == 2);
  assert(__ashrdi3(2, 1) == 1);
  assert(__lshrdi3(2, 1) == 1);
  assert(__clzsi2(1) == 31);
  assert(__clzdi2(1) == 63);
  assert(__ctzsi2(2) == 1);
  assert(__ctzdi2(2) == 1);
  assert(__ffsdi2(12) == 3);
  assert(__paritysi2(13) == 1);
  assert(__paritydi2(13) == 1);
  assert(__popcountsi2(13) == 3);
  assert(__popcountdi2(13) == 3);
  assert(__negdi2(3) == -3);
  assert(__muldi3(2,2) == 4);
  assert(__divdi3(-4,2) == -2);
  assert(__udivdi3(4,2) == 2);
  assert(__moddi3(3,2) == 1);
  assert(__umoddi3(3,2) == 1);
  assert(__udivmoddi4(5,2,&du_tmp) == 2 && du_tmp == 1);
  assert(__absvsi2(-2) == 2);
  assert(__absvdi2(-2) == 2);
  assert(__negvsi2(2) == -2);
  assert(__negvdi2(2) == -2);
  assert(__addvsi3(2, 3) == 5);
  assert(__addvdi3(2, 3) == 5);
  assert(__subvsi3(2, 3) == -1);
  assert(__subvdi3(2, 3) == -1);
  assert(__mulvsi3(2, 3) == 6);
  assert(__mulvdi3(2, 3) == 6);
  assert(__cmpdi2(3, 2) == 2);
  assert(__ucmpdi2(3, 2) == 2);
  assert(__fixsfdi(2.0) == 2);
  assert(__fixdfdi(2.0) == 2);
  assert(__fixunssfsi(2.0) == 2);
  assert(__fixunsdfsi(2.0) == 2);
  assert(__fixunssfdi(2.0) == 2);
  assert(__fixunsdfdi(2.0) == 2);
  assert(__floatdisf(2) == 2.0);
  assert(__floatdidf(2) == 2.0);
  assert(__floatundisf(2) == 2.0);
  assert(__floatundidf(2) == 2.0);
  assert(__powisf2(2.0, 2) == 4.0);
  assert(__powidf2(2.0, 2) == 4.0);

  // FIXME: Clang/LLVM seems to be miscompiling _Complex currently, probably an
  // ABI issue.
#ifndef __arm
  {
    _Complex float a = __mulsc3(1.0, 2.0, 4.0, 8.0);
    _Complex float b = (-12.0 + 16.0j);
    fprintf(stderr, "a: (%f + %f), b: (%f + %f)\n",
            __real a, __imag a, __real b, __imag b);
  }
  assert(__mulsc3(1.0, 2.0, 4.0, 8.0) == (-12.0 + 16.0j));
  assert(__muldc3(1.0, 2.0, 4.0, 8.0) == (-12.0 + 16.0j));
  assert(__divsc3(1.0, 2.0, 4.0, 8.0) == (0.25 + 0j));
  assert(__divdc3(1.0, 2.0, 4.0, 8.0) == (0.25 + 0j));
#endif

#ifdef HAS_LONG_DOUBLE
  assert(__divxc3(1.0, 2.0, 4.0, 8.0) == (0.25 + 0j));
  assert(__fixunsxfdi(2.0) == 2);
  assert(__fixunsxfsi(2.0) == 2);
  assert(__fixxfdi(2.0) == 2);
  assert(__floatdixf(2) == 2.0);
  assert(__floatundixf(2) == 2);
  assert(__mulxc3(1.0, 2.0, 4.0, 8.0) == (-12.0 + 16.0j));
  assert(__powixf2(2.0, 2) == 4.0);
#endif

  // Test some calls which are used on armv6/thumb. The calls/prototypes are
  // fake, it would be nice to test correctness, but mostly we just want to
  // make sure we resolve symbols correctly.
#if defined(__arm) && defined(__ARM_ARCH_6K__) && defined(__thumb__)
  if (argc == 100) {
    extern void __restore_vfp_d8_d15_regs(void), __save_vfp_d8_d15_regs(void);
    extern void __switch8(void), __switchu8(void),
      __switch16(void), __switch32(void);
    extern void __addsf3vfp(void);

    __addsf3vfp();
    __restore_vfp_d8_d15_regs();
    __save_vfp_d8_d15_regs();
    __switch8();
    __switchu8();
    __switch16();
    __switch32();
  }
#endif

  fprintf(stderr, "    OK!\n");

  return 0;
}
