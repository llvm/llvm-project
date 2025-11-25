////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
// 
// Copyright (c) 2014-2015, Advanced Micro Devices, Inc. All rights reserved.
// 
// Developed by:
// 
//                 AMD Research and AMD HSA Software Development
// 
//                 Advanced Micro Devices, Inc.
// 
//                 www.amd.com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// 
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

// The following set of header files provides definitions for AMD GPU
// Architecture:
//   - amd_hsa_common.h
//   - amd_hsa_elf.h
//   - amd_hsa_kernel_code.h
//   - amd_hsa_queue.h
//   - amd_hsa_signal.h
//
// Refer to "HSA Application Binary Interface: AMD GPU Architecture" for more
// information.

#ifndef AMD_HSA_COMMON_H
#define AMD_HSA_COMMON_H

#ifndef DEVICE_COMPILER
#include <stddef.h>
#include <stdint.h>
#endif

// Descriptive version of the HSA Application Binary Interface.
#define AMD_HSA_ABI_VERSION "AMD GPU Architecture v0.35 (June 25, 2015)"

// Alignment attribute that specifies a minimum alignment (in bytes) for
// variables of the specified type.
#if defined(__GNUC__) || defined(DEVICE_COMPILER)
#  define __ALIGNED__(x) __attribute__((aligned(x)))
#elif defined(_MSC_VER)
#  define __ALIGNED__(x) __declspec(align(x))
#elif defined(RC_INVOKED)
#  define __ALIGNED__(x)
#else
#  error
#endif

// Creates enumeration entries for packed types. Enumeration entries include
// bit shift amount, bit width, and bit mask.
#define AMD_HSA_BITS_CREATE_ENUM_ENTRIES(name, shift, width)                   \
  name ## _SHIFT = (shift),                                                    \
  name ## _WIDTH = (width),                                                    \
  name = (((1 << (width)) - 1) << (shift))                                     \

// Gets bits for specified mask from specified src packed instance.
#define AMD_HSA_BITS_GET(src, mask)                                            \
  ((src & mask) >> mask ## _SHIFT)                                             \

// Sets val bits for specified mask in specified dst packed instance.
#define AMD_HSA_BITS_SET(dst, mask, val)                                       \
  dst &= (~(1 << mask ## _SHIFT) & ~mask);                                     \
  dst |= (((val) << mask ## _SHIFT) & mask)                                    \

#endif // AMD_HSA_COMMON_H
