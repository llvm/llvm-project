#pragma once

// clang-format off
typedef signed char         int8_t;     // NOLINT
typedef short               int16_t;    // NOLINT
typedef long                int32_t;    // NOLINT
typedef long long           int64_t;    // NOLINT
typedef unsigned char       uint8_t;    // NOLINT
typedef unsigned short      uint16_t;   // NOLINT
typedef unsigned long       uint32_t;   // NOLINT
typedef unsigned long long  uint64_t;   // NOLINT
#ifndef _MSC_VER
typedef unsigned long long  size_t;     // NOLINT
#endif
typedef long                intptr_t;   // NOLINT
typedef unsigned long       uintptr_t;  // NOLINT
typedef long int            ptrdiff_t;  // NOLINT
typedef unsigned char       BYTE;       // NOLINT
typedef unsigned short      WORD;       // NOLINT
typedef unsigned long       DWORD;      // NOLINT
typedef int                 BOOL;       // NOLINT
typedef int                 BOOLEAN;    // NOLINT
typedef float               FLOAT;      // NOLINT
typedef int                 INT;        // NOLINT
typedef unsigned int        UINT;       // NOLINT
typedef unsigned long       ULONG;      // NOLINT
typedef short               SHORT;      // NOLINT
typedef unsigned short      USHORT;     // NOLINT
typedef char                CHAR;       // NOLINT
typedef unsigned char       UCHAR;      // NOLINT
typedef signed char         INT8;       // NOLINT
typedef signed short        INT16;      // NOLINT
typedef signed int          INT32;      // NOLINT
typedef signed long long    INT64;      // NOLINT
typedef unsigned char       UINT8;      // NOLINT
typedef unsigned short      UINT16;     // NOLINT
typedef unsigned int        UINT32;     // NOLINT
typedef unsigned long long  UINT64;     // NOLINT
typedef long                LONG;       // NOLINT
typedef signed int          LONG32;     // NOLINT
typedef unsigned int        ULONG32;    // NOLINT
typedef uint64_t            ULONG64;    // NOLINT
typedef unsigned int        DWORD32;    // NOLINT
typedef uint64_t            DWORD64;    // NOLINT
typedef uint64_t            ULONGLONG;  // NOLINT
typedef void*               PVOID;      // NOLINT
typedef void*               HANDLE;     // NOLINT
typedef void*               FILE;       // NOLINT

#define NULL                (0)         // NOLINT

#ifndef __cplusplus
typedef _Bool               bool;       // NOLINT
typedef __WCHAR_TYPE__      wchar_t;    // NOLINT
#define true                1           // NOLINT
#define false               0           // NOLINT
#endif
// clang-format on
