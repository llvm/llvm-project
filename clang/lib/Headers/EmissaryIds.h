//===- openmp/device/include/EmissaryIds.h enum & headers ----- C++ -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines Emissary API identifiers. This header is used by both host
// and device compilations.
//
//===----------------------------------------------------------------------===//

#ifndef OFFLOAD_EMISSARY_IDS_H
#define OFFLOAD_EMISSARY_IDS_H

#define __DEVATTR__
#if defined(__NVPTX__) || defined(__AMDGCN__)
#if defined(__HIP__) || defined(__CUDA__)
#if defined(__DEVATTR__)
#undef __DEVATTR__
#endif
#define __DEVATTR__ __device__
#endif
#endif

extern "C" __DEVATTR__ unsigned long long int
_emissary_exec(const unsigned long long int, ...);

#define _PACK_EMIS_IDS(a, b, c, d)                                             \
  ((unsigned long long)a << 48) | ((unsigned long long)b << 32) |              \
      ((unsigned long long)c << 16) | ((unsigned long long)d)

/// These are the various Emissary APIs currently defined.
/// MPI, HDF5, and, RESERVE are "external" Emissary APIs whose device stubs and
/// host runtime support are provided by library maintainers typically in the
/// form of a header such as "EmissaryMPI.h". The stubs call _emissary_exec.
/// The host runtime support will call functions from the actual host library
/// which are often platform specific and thus only linkable by an application.
/// A small demo of an external Emissary API (EmissaryMPI.h) is found in docs.

typedef enum {
  EMIS_ID_INVALID,
  EMIS_ID_FORTRT,
  EMIS_ID_PRINT,
  EMIS_ID_MPI,
  EMIS_ID_HDF5,
  EMIS_ID_RESERVE,
} offload_emis_id_t;

typedef enum {
  _print_INVALID,
  _printf_idx,
  _fprintf_idx,
  _ockl_asan_report_idx,
} offload_emis_print_t;

/// The future EMIS_ID_FORTRT will provide these device functions
typedef enum {
  _FortranAio_INVALID,
  _FortranAioBeginExternalListOutput_idx,
  _FortranAioOutputAscii_idx,
  _FortranAioOutputInteger32_idx,
  _FortranAioEndIoStatement_idx,
  _FortranAioOutputInteger8_idx,
  _FortranAioOutputInteger16_idx,
  _FortranAioOutputInteger64_idx,
  _FortranAioOutputReal32_idx,
  _FortranAioOutputReal64_idx,
  _FortranAioOutputComplex32_idx,
  _FortranAioOutputComplex64_idx,
  _FortranAioOutputLogical_idx,
  _FortranAAbort_idx,
  _FortranAStopStatementText_idx,
  _FortranAioBeginExternalFormattedOutput_idx,
  _FortranAStopStatement_idx,
} offload_emis_fortrt_idx;

/// This structure is created by emisExtractArgBuf to get information
/// from the data buffer passed by rpc.
typedef struct {
  unsigned int DataLen;
  unsigned int NumArgs;
  unsigned int emisid;
  unsigned int emisfnid;
  unsigned int NumSendXfers;
  unsigned int NumRecvXfers;
  unsigned long long data_not_used;
  char *keyptr;
  char *argptr;
  char *strptr;
} emisArgBuf_t;

typedef unsigned long long EmissaryReturn_t;
typedef unsigned long long emis_argptr_t;
typedef EmissaryReturn_t emisfn_t(void *, ...);

#define MAXVARGS 32

typedef enum service_rc {
  _ERC_SUCCESS = 0,
  _ERC_STATUS_ERROR = 1,
  _ERC_DATA_USED_ERROR = 2,
  _ERC_ADDINT_ERROR = 3,
  _ERC_ADDFLOAT_ERROR = 4,
  _ERC_ADDSTRING_ERROR = 5,
  _ERC_UNSUPPORTED_ID_ERROR = 6,
  _ERC_INVALID_ID_ERROR = 7,
  _ERC_ERROR_INVALID_REQUEST = 8,
  _ERC_EXCEED_MAXVARGS_ERROR = 9,
} service_rc;

#define LLVM_EMISSARY_BASE 'e'
#define LLVM_EMISSARY_OPCODE(n) (LLVM_EMISSARY_BASE << 24 | n)

typedef enum {
  OFFLOAD_EMISSARY = LLVM_EMISSARY_OPCODE(1),
  OFFLOAD_EMISSARY_DM = LLVM_EMISSARY_OPCODE(2),
} offload_emissary_t;

#endif // OFFLOAD_EMISSARY_IDS_H
