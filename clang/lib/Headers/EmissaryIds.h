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
/// The sets of emissary APIs under development
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

/// The vargs function used by emissary API device stubs
unsigned long long _emissary_exec(unsigned long long, ...);

// #define _PACK_EMIS_IDS(x, y) \
//   ((unsigned long long)x << 32) | ((unsigned long long)y)

#define _PACK_EMIS_IDS(a, b, c, d)                                             \
  ((unsigned long long)a << 48) | ((unsigned long long)b << 32) |              \
      ((unsigned long long)c << 16) | ((unsigned long long)d)

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

typedef enum service_rc {
  _ERC_SUCCESS = 0,
  _ERC_STATUS_ERROR = 1,
  _ERC_DATA_USED_ERROR = 2,
  _ERC_ADDINT_ERROR = 3,
  _ERC_ADDFLOAT_ERROR = 4,
  _ERC_ADDSTRING_ERROR = 5,
  _ERC_UNSUPPORTED_ID_ERROR = 6,
  _ERC_INVALID_ID_ERROR = 7,
  _ERC_ERROR_INVALID_REQUEST = 8
} service_rc;

#endif // OFFLOAD_EMISSARY_IDS_H
