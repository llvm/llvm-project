//===-- asan_test_utils.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
//===----------------------------------------------------------------------===//

#ifndef ASAN_TEST_UTILS_H
#define ASAN_TEST_UTILS_H

#if !defined(ASAN_EXTERNAL_TEST_CONFIG)
# define INCLUDED_FROM_ASAN_TEST_UTILS_H
# include "asan_test_config.h"
# undef INCLUDED_FROM_ASAN_TEST_UTILS_H
#endif

#include "sanitizer_test_utils.h"
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <pthread.h>
#include <stdint.h>
#include <setjmp.h>
#include <assert.h>
#include <algorithm>
#include <sys/mman.h>

#ifdef __linux__
# include <sys/prctl.h>
# include <sys/types.h>
# include <sys/stat.h>
# include <fcntl.h>
#include <unistd.h>
#endif

#ifndef __APPLE__
#include <malloc.h>
#endif

// Check that pthread_create/pthread_join return success.
#define PTHREAD_CREATE(a, b, c, d) ASSERT_EQ(0, pthread_create(a, b, c, d))
#define PTHREAD_JOIN(a, b) ASSERT_EQ(0, pthread_join(a, b))

#if ASAN_HAS_EXCEPTIONS
# define ASAN_THROW(x) throw (x)
#else
# define ASAN_THROW(x)
#endif

typedef uint8_t   U1;
typedef uint16_t  U2;
typedef uint32_t  U4;
typedef uint64_t  U8;

static const int kPageSize = 4096;

const size_t kLargeMalloc = 1 << 24;

extern void free_aaa(void *p);
extern void *malloc_aaa(size_t size);

template<typename T>
NOINLINE void asan_write(T *a) {
  *a = 0;
}

string RightOOBErrorMessage(int oob_distance, bool is_write);
string RightOOBWriteMessage(int oob_distance);
string RightOOBReadMessage(int oob_distance);
string LeftOOBErrorMessage(int oob_distance, bool is_write);
string LeftOOBWriteMessage(int oob_distance);
string LeftOOBReadMessage(int oob_distance);
string LeftOOBAccessMessage(int oob_distance);
char* MallocAndMemsetString(size_t size, char ch);
char* MallocAndMemsetString(size_t size);

extern char glob1[1];
extern char glob2[2];
extern char glob3[3];
extern char glob4[4];
extern char glob5[5];
extern char glob6[6];
extern char glob7[7];
extern char glob8[8];
extern char glob9[9];
extern char glob10[10];
extern char glob11[11];
extern char glob12[12];
extern char glob13[13];
extern char glob14[14];
extern char glob15[15];
extern char glob16[16];
extern char glob17[17];
extern char glob1000[1000];
extern char glob10000[10000];
extern char glob100000[100000];
extern int GlobalsTest(int x);

#endif  // ASAN_TEST_UTILS_H
