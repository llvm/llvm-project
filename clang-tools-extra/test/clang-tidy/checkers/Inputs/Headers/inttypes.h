//===--- inttypes.h - Stub header for tests ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _INTTYPES_H_
#define _INTTYPES_H_

typedef __INTMAX_TYPE__  intmax_t;
typedef __INT64_TYPE__ int64_t;
typedef __INT32_TYPE__ int32_t;
typedef __INT16_TYPE__ int16_t;
typedef __INT8_TYPE__ int8_t;

typedef __UINTMAX_TYPE__ uintmax_t;
typedef __UINT64_TYPE__ uint64_t;
typedef __UINT32_TYPE__ uint32_t;
typedef __UINT16_TYPE__ uint16_t;
typedef __UINT8_TYPE__ uint8_t;

#define PRIdMAX "lld"
#define PRId64 "lld"
#define PRId32 "d"
#define PRId16 "hd"
#define PRId8  "hhd"

#define PRIiMAX "lli"
#define PRIi64 "lli"
#define PRIi32 "i"
#define PRIi16 "hi"
#define PRIi8  "hhi"

#define PRIiFAST64 "lli"
#define PRIiFAST32 "i"
#define PRIiFAST16 "hi"
#define PRIiFAST8  "hhi"

#define PRIiLEAST64 "lli"
#define PRIiLEAST32 "i"
#define PRIiLEAST16 "hi"
#define PRIiLEAST8  "hhi"

#define PRIuMAX "llu"
#define PRIu64 "llu"
#define PRIu32 "u"
#define PRIu16 "hu"
#define PRIu8  "hhu"

#define PRIuFAST64 "llu"
#define PRIuFAST32 "u"
#define PRIuFAST16 "hu"
#define PRIuFAST8  "hhu"

#define PRIuLEAST64 "llu"
#define PRIuLEAST32 "u"
#define PRIuLEAST16 "hu"
#define PRIuLEAST8  "hhu"

#endif // _INTTYPES_H_
