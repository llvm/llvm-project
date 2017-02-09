/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#ifndef DEVICE_AMD_HSA_H
#define DEVICE_AMD_HSA_H

typedef char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long int64_t;
typedef unsigned long uint64_t;

#ifdef __LP64__
#undef __LP64__
#endif
#define __LP64__
#define DEVICE_COMPILER
#define LITTLEENDIAN_CPU
#include "hsa.h"
#include "amd_hsa_common.h"
#include "amd_hsa_elf.h"
#include "amd_hsa_kernel_code.h"
#include "amd_hsa_queue.h"
#include "amd_hsa_signal.h"
#include "device_amd_hsa.h"
#undef DEVICE_COMPILER

#endif // DEVICE_AMD_HSA_H
