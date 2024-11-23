/*
 * Copyright (c) 2020 Bitdefender
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef BDDISASM_TYPES_H
#define BDDISASM_TYPES_H


#if defined(_MSC_VER) || defined(__ICC) || defined(__INTEL_COMPILER)

// Microsoft VC compiler.

typedef unsigned __int8 ND_UINT8;
typedef unsigned __int16 ND_UINT16;
typedef unsigned __int32 ND_UINT32;
typedef unsigned __int64 ND_UINT64;
typedef signed __int8 ND_SINT8;
typedef signed __int16 ND_SINT16;
typedef signed __int32 ND_SINT32;
typedef signed __int64 ND_SINT64;

#elif defined(__GNUC__) || defined(__GNUG__) || defined(__clang__)

// clang/GCC compiler.

typedef __UINT8_TYPE__ ND_UINT8;
typedef __UINT16_TYPE__ ND_UINT16;
typedef __UINT32_TYPE__ ND_UINT32;
typedef __UINT64_TYPE__ ND_UINT64;
typedef __INT8_TYPE__ ND_SINT8;
typedef __INT16_TYPE__ ND_SINT16;
typedef __INT32_TYPE__ ND_SINT32;
typedef __INT64_TYPE__ ND_SINT64;

#else

// other compilers, assume stdint is present.

#include <stdint.h>

typedef uint8_t ND_UINT8;
typedef uint16_t ND_UINT16;
typedef uint32_t ND_UINT32;
typedef uint64_t ND_UINT64;
typedef int8_t ND_SINT8;
typedef int16_t ND_SINT16;
typedef int32_t ND_SINT32;
typedef int64_t ND_SINT64;

#endif



#if defined(_M_AMD64) || defined(__x86_64__)

#define ND_ARCH_X64

#elif defined(_M_IX86) || defined(__i386__)

#define ND_ARCH_X86
#define ND_ARCH_IA32

#elif defined(_M_ARM64) || defined(__aarch64__)

#define ND_ARCH_AARCH64
#define ND_ARCH_A64

#elif defined(_M_ARM) || defined(__arm__)

#define ND_ARCH_ARM
#define ND_ARCH_A32

#else

#error "Unknown architecture!"

#endif


// Handle architecture definitions.
#if defined(ND_ARCH_X64) || defined(ND_ARCH_A64)

typedef ND_UINT64 ND_SIZET;

#elif defined(ND_ARCH_X86) || defined(ND_ARCH_A32)

typedef ND_UINT32 ND_SIZET;

#else

#error "Unknown architecture!"

#endif


// Common definitions.
typedef ND_UINT8 ND_BOOL;

#if defined(__cplusplus)
#define ND_NULL     nullptr
#else
#define ND_NULL     ((void *)(0))
#endif
#define ND_TRUE     (1)
#define ND_FALSE    (0)


#endif // BDDISASM_TYPES_H
