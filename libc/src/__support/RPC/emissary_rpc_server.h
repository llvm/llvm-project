//===-- Shared memory RPC server instantiation ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is intended to be used externally as part of the `shared/`
// interface. Consider this an extenion of rpc_server.h to support emissary
// APIs. rpc_server.h must be included first.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_RPC_EMISSARY_RPC_SERVER_H
#define LLVM_LIBC_SRC___SUPPORT_RPC_EMISSARY_RPC_SERVER_H

#include "../clang/lib/Headers/EmissaryIds.h"
#include "rpc_server.h"
#include <string.h>
#include <unordered_map>

namespace EmissaryExternal {
extern "C" {
/// Called by EmissaryTop for all MPI emissary API functions
__attribute((weak)) EmissaryReturn_t EmissaryMPI(char *data, emisArgBuf_t *ab,
                                                 emis_argptr_t *arg[]);
/// Called by EmissaryTop for all HDF5 Emissary API functions
__attribute((weak)) EmissaryReturn_t EmissaryHDF5(char *data, emisArgBuf_t *ab,
                                                  emis_argptr_t *arg[]);
/// Called by EmissaryTop to support user-defined emissary API
__attribute((weak)) EmissaryReturn_t EmissaryReserve(char *data,
                                                     emisArgBuf_t *ab,
                                                     emis_argptr_t *arg[]);
/// Called by EmissaryTop to support Fortran IO runtime
__attribute((weak)) EmissaryReturn_t EmissaryFortrt(char *data,
                                                    emisArgBuf_t *ab,
                                                    emis_argptr_t *arg[]);
} // end extern "C"
} // namespace EmissaryExternal

// We would like to get llvm typeID enum from Type.h. e.g.
// #include ".../llvm/include/llvm/IR/Type.h"
// But we cannot include LLVM headers in a runtime function.
// So we a have a manual copy of llvm TypeID enum from Type.h
// The codegen for _emissary_exec puts this ID in the key for
// each arg and the host runtime needs to decode this key.
enum TypeID {
  // PrimitiveTypes
  HalfTyID = 0,  ///< 16-bit floating point type
  BFloatTyID,    ///< 16-bit floating point type (7-bit significand)
  FloatTyID,     ///< 32-bit floating point type
  DoubleTyID,    ///< 64-bit floating point type
  X86_FP80TyID,  ///< 80-bit floating point type (X87)
  FP128TyID,     ///< 128-bit floating point type (112-bit significand)
  PPC_FP128TyID, ///< 128-bit floating point type (two 64-bits, PowerPC)
  VoidTyID,      ///< type with no size
  LabelTyID,     ///< Labels
  MetadataTyID,  ///< Metadata
  X86_AMXTyID,   ///< AMX vectors (8192 bits, X86 specific)
  TokenTyID,     ///< Tokens

  // Derived types... see DerivedTypes.h file.
  IntegerTyID,        ///< Arbitrary bit width integers
  ByteTyID,           ///< Arbitrary bit width bytes
  FunctionTyID,       ///< Functions
  PointerTyID,        ///< Pointers
  StructTyID,         ///< Structures
  ArrayTyID,          ///< Arrays
  FixedVectorTyID,    ///< Fixed width SIMD vector type
  ScalableVectorTyID, ///< Scalable SIMD vector type
  TypedPointerTyID,   ///< Typed pointer used by some GPU targets
  TargetExtTyID,      ///< Target extension type
};

// -----  Begin support for EmissaryPrint -----

namespace LIBC_NAMESPACE_DECL {
namespace internal {

// Server Support for EmissaryPrint starts here

// NUMFPREGS and FPREGSZ are part of x86 vargs ABI that
// is recreated with this printf support.
#define NUMFPREGS 8
#define FPREGSZ 16

typedef int uint128_t __attribute__((mode(TI)));
struct emissary_pfIntRegs {
  uint64_t rdi, rsi, rdx, rcx, r8, r9;
};
typedef struct emissary_pfIntRegs emissary_pfIntRegs_t; // size = 48 bytes

struct emissary_pfRegSaveArea {
  emissary_pfIntRegs_t iregs;
  uint128_t freg[NUMFPREGS];
};
typedef struct emissary_pfRegSaveArea
    emissary_pfRegSaveArea_t; // size = 304 bytes

struct emissary_ValistExt {
  uint32_t gp_offset;      /* offset to next available gpr in reg_save_area */
  uint32_t fp_offset;      /* offset to next available fpr in reg_save_area */
  void *overflow_arg_area; /* args that are passed on the stack */
  emissary_pfRegSaveArea_t *reg_save_area; /* int and fp registers */
  size_t overflow_size;
} __attribute__((packed));
typedef struct emissary_ValistExt emissary_ValistExt_t;

// Handle overflow when building the va_list for vprintf
static service_rc emissary_pfGetOverflow(emissary_ValistExt_t *valist,
                                         size_t needsize) {
  if (needsize < valist->overflow_size)
    return _ERC_SUCCESS;

  // Make the overflow area bigger
  size_t stacksize;
  void *newstack;
  if (valist->overflow_size == 0) {
    // Make initial save area big to reduce mallocs
    stacksize = (FPREGSZ * NUMFPREGS) * 2;
    if (needsize > stacksize)
      stacksize = needsize; // maybe a big string
  } else {
    // Initial save area not big enough, double it
    stacksize = valist->overflow_size * 2;
  }
  if (!(newstack = malloc(stacksize))) {
    return _ERC_STATUS_ERROR;
  }
  memset(newstack, 0, stacksize);
  if (valist->overflow_size) {
    memcpy(newstack, valist->overflow_arg_area, valist->overflow_size);
    free(valist->overflow_arg_area);
  }
  valist->overflow_arg_area = newstack;
  valist->overflow_size = stacksize;
  return _ERC_SUCCESS;
}

// Add an integer to the va_list for vprintf
static service_rc emissary_pfAddInteger(emissary_ValistExt_t *valist, char *val,
                                        size_t valsize, size_t *stacksize) {
  uint64_t ival;
  switch (valsize) {
  case 1:
    ival = *(uint8_t *)val;
    break;
  case 2:
    ival = *(uint32_t *)val;
    break;
  case 4:
    ival = (*(uint32_t *)val);
    break;
  case 8:
    ival = *(uint64_t *)val;
    break;
  default: {
    return _ERC_STATUS_ERROR;
  }
  }
  //  Always copy 8 bytes, sizeof(ival)
  if ((valist->gp_offset + sizeof(ival)) <= sizeof(emissary_pfIntRegs_t)) {
    memcpy(((char *)valist->reg_save_area + valist->gp_offset), &ival,
           sizeof(ival));
    valist->gp_offset += sizeof(ival);
    return _ERC_SUCCESS;
  }
  // Ensure valist overflow area is big enough
  size_t needsize = (size_t)*stacksize + sizeof(ival);
  if (emissary_pfGetOverflow(valist, needsize) != _ERC_SUCCESS)
    return _ERC_STATUS_ERROR;
  // Copy to overflow
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, &ival,
         sizeof(ival));

  *stacksize += sizeof(ival);
  return _ERC_SUCCESS;
}

// Add a String argument when building va_list for vprintf
static service_rc emissary_pfAddString(emissary_ValistExt_t *valist, char *val,
                                       size_t strsz, size_t *stacksize) {
  size_t valsize =
      sizeof(char *); // ABI captures pointer to string,  not string
  if ((valist->gp_offset + valsize) <= sizeof(emissary_pfIntRegs_t)) {
    memcpy(((char *)valist->reg_save_area + valist->gp_offset), val, valsize);
    valist->gp_offset += valsize;
    return _ERC_SUCCESS;
  }
  size_t needsize = (size_t)*stacksize + valsize;
  if (emissary_pfGetOverflow(valist, needsize) != _ERC_SUCCESS)
    return _ERC_STATUS_ERROR;
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, val,
         valsize);
  *stacksize += valsize;
  return _ERC_SUCCESS;
}

// Add a floating point value when building va_list for vprintf
static service_rc emissary_pfAddFloat(emissary_ValistExt_t *valist,
                                      char *numdata, size_t valsize,
                                      size_t *stacksize) {
  // we could use load because doubles are now aligned
  double dval;
  if (valsize == 4) {
    float fval;
    memcpy(&fval, numdata, 4);
    dval = (double)fval; // Extend single to double per abi
  } else if (valsize == 8) {
    memcpy(&dval, numdata, 8);
  } else {
    return _ERC_STATUS_ERROR;
  }
  if ((valist->fp_offset + FPREGSZ) <= sizeof(emissary_pfRegSaveArea_t)) {
    memcpy(((char *)valist->reg_save_area + (size_t)(valist->fp_offset)), &dval,
           sizeof(double));
    valist->fp_offset += FPREGSZ;
    return _ERC_SUCCESS;
  }
  size_t needsize = (size_t)*stacksize + sizeof(double);
  if (emissary_pfGetOverflow(valist, needsize) != _ERC_SUCCESS)
    return _ERC_STATUS_ERROR;
  memcpy((char *)(valist->overflow_arg_area) + (size_t)*stacksize, &dval,
         sizeof(double));
  // move only by the size of the double (8 bytes)
  *stacksize += sizeof(double);
  return _ERC_SUCCESS;
}

// Build an extended va_list for vprintf by unpacking the buffer
static service_rc emissary_pfBuildValist(emissary_ValistExt_t *valist,
                                         int NumArgs, char *keyptr,
                                         char *dataptr, char *strptr,
                                         unsigned long long *data_not_used) {
  emissary_pfRegSaveArea_t *regs;
  size_t regs_size = sizeof(*regs);
  regs = (emissary_pfRegSaveArea_t *)malloc(regs_size);
  if (!regs)
    return _ERC_STATUS_ERROR;
  memset(regs, 0, regs_size);
  *valist = (emissary_ValistExt_t){
      .gp_offset = 0,
      .fp_offset = 0,
      .overflow_arg_area = NULL,
      .reg_save_area = regs,
      .overflow_size = 0,
  };

  size_t num_bytes;
  size_t bytes_consumed;
  size_t strsz;
  size_t fillerNeeded;

  size_t stacksize = 0;

  for (int argnum = 0; argnum < NumArgs; argnum++) {
    num_bytes = 0;
    strsz = 0;
    unsigned int key = *(unsigned int *)keyptr;
    unsigned int llvmID = key >> 16;
    unsigned int numbits = (key << 16) >> 16;
    // fprintf(stderr, " key:%d llvmId:%d numbits:%d arg:%d of %d  PID:%d\n",
    // key,   llvmID,    numbits,   argnum, NumArgs, PointerTyID);
    switch (llvmID) {
    case FloatTyID:  ///<  2: 32-bit floating point type
    case DoubleTyID: ///<  3: 64-bit floating point type
    case FP128TyID:  ///<  5: 128-bit floating point type (112-bit mantissa)
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return _ERC_DATA_USED_ERROR;
      if (valist->fp_offset == 0)
        valist->fp_offset = sizeof(emissary_pfIntRegs_t);
      if (emissary_pfAddFloat(valist, dataptr, num_bytes, &stacksize))
        return _ERC_ADDFLOAT_ERROR;
      break;

    case IntegerTyID: ///< 11: Arbitrary bit width integers
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return _ERC_DATA_USED_ERROR;
      if (emissary_pfAddInteger(valist, dataptr, num_bytes, &stacksize))
        return _ERC_ADDINT_ERROR;
      break;

    case PointerTyID:     ///< 15: Pointers
      if (numbits == 1) { // This is a pointer to string
        num_bytes = 4;
        bytes_consumed = num_bytes;
        strsz = (size_t)*(unsigned int *)dataptr;
        if ((*data_not_used) < bytes_consumed)
          return _ERC_DATA_USED_ERROR;
        if (strsz == 0) {
          if (emissary_pfAddInteger(valist, dataptr, num_bytes, &stacksize))
            return _ERC_ADDINT_ERROR;
        } else {
          if (emissary_pfAddString(valist, (char *)&strptr, strsz, &stacksize))
            return _ERC_ADDSTRING_ERROR;
        }
      } else {
        num_bytes = 8;
        bytes_consumed = num_bytes;
        fillerNeeded = ((size_t)dataptr) % num_bytes;
        if (fillerNeeded) {
          dataptr += fillerNeeded; // dataptr is now aligned
          bytes_consumed += fillerNeeded;
        }
        if ((*data_not_used) < bytes_consumed)
          return _ERC_DATA_USED_ERROR;
        if (emissary_pfAddInteger(valist, dataptr, num_bytes, &stacksize))
          return _ERC_ADDINT_ERROR;
      }
      break;

    case HalfTyID:           ///<  1: 16-bit floating point type
    case ArrayTyID:          ///< 14: Arrays
    case StructTyID:         ///< 13: Structures
    case FunctionTyID:       ///< 12: Functions
    case TokenTyID:          ///< 10: Tokens
    case MetadataTyID:       ///<  8: Metadata
    case LabelTyID:          ///<  7: Labels
    case PPC_FP128TyID:      ///<  6: 128-bit floating point type (two 64-bits,
                             ///<  PowerPC)
    case X86_FP80TyID:       ///<  4: 80-bit floating point type (X87)
    case FixedVectorTyID:    ///< 16: Fixed width SIMD vector type
    case ScalableVectorTyID: ///< 17: Scalable SIMD vector type
    case TypedPointerTyID:   ///< Typed pointer used by some GPU targets
    case TargetExtTyID:      ///< Target extension type
    case VoidTyID:
      return _ERC_UNSUPPORTED_ID_ERROR;
      break;
    default:
      return _ERC_INVALID_ID_ERROR;
    }

    dataptr += num_bytes;
    strptr += strsz;
    *data_not_used -= bytes_consumed;
    keyptr += 4;
  }
  return _ERC_SUCCESS;
} // end emissary_pfBuildValist

static service_rc emissary_printf(uint *rc, emisArgBuf_t *ab) {
  if (ab->DataLen == 0)
    return _ERC_SUCCESS;

  char *fmtstr = ab->strptr;

  // Skip past the format string
  ab->NumArgs--;
  ab->keyptr += 4;
  size_t abstrsz = (size_t)*(unsigned int *)ab->argptr;

  ab->strptr += abstrsz;
  ab->argptr += 4;
  ab->data_not_used -= 4;

  emissary_ValistExt_t valist; // FIXME: We may need to align this declare
  va_list *real_va_list;
  real_va_list = (va_list *)&valist;

  if (emissary_pfBuildValist(&valist, ab->NumArgs, ab->keyptr, ab->argptr,
                             ab->strptr, &ab->data_not_used) != _ERC_SUCCESS)
    return _ERC_ERROR_INVALID_REQUEST;

  // Roll back offsets and save stack pointer for
  valist.gp_offset = 0;
  valist.fp_offset = sizeof(emissary_pfIntRegs_t);
  void *save_stack = valist.overflow_arg_area;
  *rc = vprintf(fmtstr, *real_va_list);
  if (valist.reg_save_area)
    free(valist.reg_save_area);
  if (save_stack)
    free(save_stack);
  return _ERC_SUCCESS;
}

// emisExtractArgBuf extract ArgBuf using protocol EmitEmissaryExec makes.
static void emisExtractArgBuf(char *data, emisArgBuf_t *ab) {

  uint32_t *int32_data = (uint32_t *)data;
  ab->DataLen = int32_data[0];
  ab->NumArgs = int32_data[1];

  // Note: while the data buffer contains all args including strings,
  // ab->DataLen does not include strings. It only counts header, keys,
  // and aligned numerics.

  ab->keyptr = data + (2 * sizeof(int));
  ab->argptr = ab->keyptr + (ab->NumArgs * sizeof(int));
  ab->strptr = data + (size_t)ab->DataLen;
  int alignfill = 0;
  if (((size_t)ab->argptr) % (size_t)8) {
    ab->argptr += 4;
    alignfill = 4;
  }

  // Extract the two emissary identifiers and number of send
  // and recv device data transfers. These are 4 16 bit values
  // packed into a single 64-bit field.
  uint64_t arg1 = *(uint64_t *)ab->argptr;
  ab->emisid = (unsigned int)((arg1 >> 48) & 0xFFFF);
  ab->emisfnid = (unsigned int)((arg1 >> 32) & 0xFFFF);
  ab->NumSendXfers = (unsigned int)((arg1 >> 16) & 0xFFFF);
  ab->NumRecvXfers = (unsigned int)((arg1) & 0xFFFF);

  // skip the uint64_t emissary id arg which is first arg in _emissary_exec.
  ab->keyptr += sizeof(int);
  ab->argptr += sizeof(uint64_t);
  ab->NumArgs -= 1;

  // data_not_used used for testing consistency.
  ab->data_not_used =
      (size_t)(ab->DataLen) - (((size_t)(3 + ab->NumArgs) * sizeof(int)) +
                               sizeof(uint64_t) + alignfill);

  // Ensure first arg after emissary id arg is aligned.
  if (((size_t)ab->argptr) % (size_t)8) {
    ab->argptr += 4;
    ab->data_not_used -= 4;
  }
}

/// Get uint32 value extended to uint64_t value from a char ptr
static uint64_t getuint32(char *val) {
  uint32_t i32 = *(uint32_t *)val;
  return (uint64_t)i32;
}

/// Get uint64_t value from a char ptr
static uint64_t getuint64(char *val) { return *(uint64_t *)val; }

// ------------------ Utils -----------------------------------------

// build argument array to create call to variadic wrappers
static uint32_t
EmissaryBuildVargs(int NumArgs, char *keyptr, char *dataptr, char *strptr,
                   unsigned long long *data_not_used, emis_argptr_t *a[],
                   std::unordered_map<void *, void *> *D2HAddrList) {
  size_t num_bytes;
  size_t bytes_consumed;
  size_t strsz;
  size_t fillerNeeded;

  uint argcount = 0;

  for (int argnum = 0; argnum < NumArgs; argnum++) {
    num_bytes = 0;
    strsz = 0;
    unsigned int key = *(unsigned int *)keyptr;
    unsigned int llvmID = key >> 16;
    unsigned int numbits = (key << 16) >> 16;

    switch (llvmID) {
    case FloatTyID:  ///<  2: 32-bit floating point type
    case DoubleTyID: ///<  3: 64-bit floating point type
    case FP128TyID:  ///<  5: 128-bit floating point type (112-bit mantissa)
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return _ERC_DATA_USED_ERROR;

      if (num_bytes == 4)
        a[argcount] = (emis_argptr_t *)getuint32(dataptr);
      else
        a[argcount] = (emis_argptr_t *)getuint64(dataptr);
      break;

    case IntegerTyID: ///< 11: Arbitrary bit width integers
      num_bytes = numbits / 8;
      bytes_consumed = num_bytes;
      fillerNeeded = ((size_t)dataptr) % num_bytes;
      if (fillerNeeded) {
        dataptr += fillerNeeded;
        bytes_consumed += fillerNeeded;
      }
      if ((*data_not_used) < bytes_consumed)
        return _ERC_DATA_USED_ERROR;

      if (num_bytes == 4)
        a[argcount] = (emis_argptr_t *)getuint32(dataptr);
      else
        a[argcount] = (emis_argptr_t *)getuint64(dataptr);
      break;

    case PointerTyID: {   ///< 15: Pointers
      if (numbits == 1) { // This is a pointer to string
        num_bytes = 4;
        bytes_consumed = num_bytes;
        strsz = (size_t)*(unsigned int *)dataptr;
        if ((*data_not_used) < bytes_consumed)
          return _ERC_DATA_USED_ERROR;
        a[argcount] = (emis_argptr_t *)((char *)strptr);
      } else {
        num_bytes = 8;
        bytes_consumed = num_bytes;
        fillerNeeded = ((size_t)dataptr) % num_bytes;
        if (fillerNeeded) {
          dataptr += fillerNeeded; // dataptr is now aligned
          bytes_consumed += fillerNeeded;
        }
        if ((*data_not_used) < bytes_consumed)
          return _ERC_DATA_USED_ERROR;
        a[argcount] = (emis_argptr_t *)getuint64(dataptr);
      }
      if (D2HAddrList) {
        auto found = D2HAddrList->find((void *)a[argcount]);
        if (found != D2HAddrList->end())
          a[argcount] = (emis_argptr_t *)found->second;
      }
    } break;

    case HalfTyID:           ///<  1: 16-bit floating point type
    case ArrayTyID:          ///< 14: Arrays
    case StructTyID:         ///< 13: Structures
    case FunctionTyID:       ///< 12: Functions
    case TokenTyID:          ///< 10: Tokens
    case MetadataTyID:       ///<  8: Metadata
    case LabelTyID:          ///<  7: Labels
    case PPC_FP128TyID:      ///<  6: 128-bit floating point type (two 64-bits,
                             ///<  PowerPC)
    case X86_FP80TyID:       ///<  4: 80-bit floating point type (X87)
    case FixedVectorTyID:    ///< 16: Fixed width SIMD vector type
    case ScalableVectorTyID: ///< 17: Scalable SIMD vector type
    case TypedPointerTyID:   ///< Typed pointer used by some GPU targets
    case TargetExtTyID:      ///< Target extension type
    case VoidTyID:
      return _ERC_UNSUPPORTED_ID_ERROR;
      break;
    default:
      return _ERC_INVALID_ID_ERROR;
    }
    // Move to next argument
    dataptr += num_bytes;
    strptr += strsz;
    *data_not_used -= bytes_consumed;
    keyptr += 4;
    argcount++;
  }
  return _ERC_SUCCESS;
}

//  Utility to skip two args in the ArgBuf
static void emisSkipXferArgSet(emisArgBuf_t *ab) {
  // Skip the ptr and size of the Xfer
  ab->NumArgs -= 2;
  ab->keyptr += 2 * sizeof(uint32_t);
  ab->argptr += 2 * sizeof(void *);
  ab->data_not_used -= 2 * sizeof(void *);
}

static service_rc emissary_fprintf(uint *rc, emisArgBuf_t *ab) {

  if (ab->DataLen == 0)
    return _ERC_SUCCESS;
  char *fmtstr = ab->strptr;
  FILE *fileptr = (FILE *)*((size_t *)ab->argptr);

  // Skip past the file pointer
  ab->NumArgs--;
  ab->keyptr += 4;
  ab->argptr += sizeof(FILE *);
  ab->data_not_used -= sizeof(FILE *);

  // Skip past the format string
  ab->NumArgs--;
  ab->keyptr += 4;
  size_t abstrsz = (size_t)*(unsigned int *)ab->argptr;
  ab->strptr += abstrsz;
  ab->argptr += 4;
  ab->data_not_used -= 4;

  emissary_ValistExt_t valist; // FIXME: we may want to align this declare
  va_list *real_va_list;
  real_va_list = (va_list *)&valist;

  if (emissary_pfBuildValist(&valist, ab->NumArgs, ab->keyptr, ab->argptr,
                             ab->strptr, &ab->data_not_used) != _ERC_SUCCESS)
    return _ERC_ERROR_INVALID_REQUEST;

  // Roll back offsets and save stack pointer
  valist.gp_offset = 0;
  valist.fp_offset = sizeof(emissary_pfIntRegs_t);
  void *save_stack = valist.overflow_arg_area;
  *rc = vfprintf(fileptr, fmtstr, *real_va_list);
  if (valist.reg_save_area)
    free(valist.reg_save_area);
  if (save_stack)
    free(save_stack);
  return _ERC_SUCCESS;
}

static EmissaryReturn_t EmissaryPrint(char *data, emisArgBuf_t *ab) {
  uint32_t return_value;
  service_rc rc;
  switch (ab->emisfnid) {
  case _printf_idx: {
    rc = emissary_printf(&return_value, ab);
    break;
  }
  case _fprintf_idx: {
    rc = emissary_fprintf(&return_value, ab);
    break;
  }
  case _ockl_asan_report_idx: {
    fprintf(stderr, " asan_report not yet implemented\n");
    return_value = 0;
    rc = _ERC_STATUS_ERROR;
    break;
  }
  case _print_INVALID:
  default: {
    fprintf(stderr, " INVALID emissary function id (%d) for PRINT API \n",
            ab->emisfnid);
    return_value = 0;
    rc = _ERC_STATUS_ERROR;
    break;
  }
  }
  if (rc != _ERC_SUCCESS)
    fprintf(stderr, "HOST failure in _emissary_execute_print rc:%d\n", rc);

  return (EmissaryReturn_t)return_value;
}

static EmissaryReturn_t
EmissaryTop(char *data, emisArgBuf_t *ab,
            std::unordered_map<void *, void *> *D2HAddrList) {
  EmissaryReturn_t result = 0;
  emis_argptr_t **args = (emis_argptr_t **)aligned_alloc(
      sizeof(emis_argptr_t), ab->NumArgs * sizeof(emis_argptr_t *));

  switch (ab->emisid) {
  case EMIS_ID_INVALID: {
    fprintf(stderr, "Emissary (host execution) got invalid EMIS_ID\n");
    result = 0;
    break;
  }
  case EMIS_ID_PRINT: {
    result = EmissaryPrint(data, ab);
    break;
  }
  case EMIS_ID_MPI: {
    if (EmissaryBuildVargs(ab->NumArgs, ab->keyptr, ab->argptr, ab->strptr,
                           &(ab->data_not_used), &args[0],
                           D2HAddrList) != _ERC_SUCCESS) {
      return (EmissaryReturn_t)0;
    }
    result = EmissaryExternal::EmissaryMPI(data, ab, args);
    break;
  }
  case EMIS_ID_HDF5: {
    if (EmissaryBuildVargs(ab->NumArgs, ab->keyptr, ab->argptr, ab->strptr,
                           &(ab->data_not_used), &args[0],
                           D2HAddrList) != _ERC_SUCCESS)
      return (EmissaryReturn_t)0;
    result = EmissaryExternal::EmissaryHDF5(data, ab, args);
    break;
  }
  case EMIS_ID_FORTRT: {
    if (EmissaryBuildVargs(ab->NumArgs, ab->keyptr, ab->argptr, ab->strptr,
                           &(ab->data_not_used), &args[0],
                           D2HAddrList) != _ERC_SUCCESS)
      return (EmissaryReturn_t)0;
    result = EmissaryExternal::EmissaryFortrt(data, ab, args);
    break;
    break;
  }

  case EMIS_ID_RESERVE: {
    if (EmissaryBuildVargs(ab->NumArgs, ab->keyptr, ab->argptr, ab->strptr,
                           &(ab->data_not_used), &args[0],
                           D2HAddrList) != _ERC_SUCCESS)
      return (EmissaryReturn_t)0;
    result = EmissaryExternal::EmissaryReserve(data, ab, args);
    break;
  }
  default:
    fprintf(stderr,
            "Emissary (host execution) EMIS_ID:%d fnid:%d not supported\n",
            ab->emisid, ab->emisfnid);
  }
  free(args);
  return result;
}

// -----------------------------------------------------------------
// -- Handle OFFLOAD_EMISSARY and OFFLOAD_EMISSARY_DM opcodes     --
// -- handle_emissary_impl calls EmissaryTop for each active lane --
// -----------------------------------------------------------------
template <uint32_t NumLanes>
LIBC_INLINE static ::rpc::Status
handle_emissary_impl(::rpc::Server::Port &port) {

  switch (port.get_opcode()) {

  // This case handles the device function __llvm_emissary_rpc for emissary
  // APIs that require no d2h or h2d memory transfer.
  case OFFLOAD_EMISSARY: {
    uint64_t Sizes[NumLanes] = {0};
    unsigned long long Results[NumLanes] = {0};
    void *buf_ptrs[NumLanes] = {nullptr};
    port.recv_n(buf_ptrs, Sizes, [&](uint64_t Size) { return new char[Size]; });
    uint32_t id = 0;
    for (void *buffer_ptr : buf_ptrs) {
      if (buffer_ptr) {
        emisArgBuf_t ab;
        emisExtractArgBuf((char *)buffer_ptr, &ab);
        Results[id++] = EmissaryTop((char *)buffer_ptr, &ab, nullptr);
      }
    }
    port.send([&](::rpc::Buffer *Buffer, uint32_t ID) {
      Buffer->data[0] = static_cast<uint64_t>(Results[ID]);
      delete[] reinterpret_cast<char *>(buf_ptrs[ID]);
    });
    break;
  }

  // This case handles the device function __llvm_emissary_rpc_dm for emissary
  // APIs require D2H or H2D transfer vectors to be processed through the port.
  // FIXME: test with multiple transfer vectors of the same type.
  case OFFLOAD_EMISSARY_DM: {
    uint64_t Sizes[NumLanes] = {0};
    unsigned long long Results[NumLanes] = {0};
    void *buf_ptrs[NumLanes] = {nullptr};
    port.recv_n(buf_ptrs, Sizes, [&](uint64_t Size) { return new char[Size]; });

    uint32_t id = 0;
    emisArgBuf_t AB[NumLanes];
    std::unordered_map<void *, void *> D2HAddrList;
    void *Xfers[NumLanes] = {nullptr};
    void *devXfers[NumLanes] = {nullptr};
    uint64_t XferSzs[NumLanes] = {0};
    uint32_t numSendXfers = 0;
    id = 0;

    for (void *buffer_ptr : buf_ptrs) {
      if (buffer_ptr) {

        emisArgBuf_t *ab = &AB[id];
        emisExtractArgBuf((char *)buffer_ptr, ab);
        for (uint32_t idx = 0; idx < ab->NumSendXfers; idx++) {
          numSendXfers++;
          devXfers[id] = (void *)*((uint64_t *)ab->argptr);
          XferSzs[id] = (size_t)*((size_t *)(ab->argptr + sizeof(void *)));
          emisSkipXferArgSet(ab);
        }
        // Allocate the host space for the receive Xfers
        for (uint32_t idx = 0; idx < ab->NumRecvXfers; idx++) {
          void *devAddr = (void *)*((uint64_t *)ab->argptr);
          size_t devSz = (((size_t)*((size_t *)(ab->argptr + sizeof(void *)))) &
                          0x00000000FFFFFFFF);
          void *hostAddr = new char[devSz];
          D2HAddrList.insert(std::pair<void *, void *>(devAddr, hostAddr));
          emisSkipXferArgSet(ab);
        }
        id++;
      }
    }

    // recv_n for device send_n into new host-allocated Xfers
    if (numSendXfers)
      port.recv_n(Xfers, XferSzs,
                  [&](uint64_t Size) { return new char[Size]; });

    // Xfers now contains just allocated host addrs for sends and
    // devXfers contains corresponding devAddr for those sends
    // Build map to pass to Emissary
    id = 0;
    for (void *Xfer : Xfers) {
      if (Xfer) {
        D2HAddrList.insert(std::pair<void *, void *>(devXfers[id], Xfer));
        id++;
      }
    }

    // Call EmissaryTop for each active lane
    id = 0;
    for (void *buffer_ptr : buf_ptrs) {
      if (buffer_ptr) {
        emisArgBuf_t *ab = &AB[id];
        emisExtractArgBuf((char *)buffer_ptr, ab);
        for (uint32_t idx = 0; idx < ab->NumSendXfers; idx++)
          emisSkipXferArgSet(ab);
        for (uint32_t idx = 0; idx < ab->NumRecvXfers; idx++)
          emisSkipXferArgSet(ab);
        Results[id] = EmissaryTop((char *)buffer_ptr, ab, &D2HAddrList);
        id++;
      }
    }

    // Process send_n for the H2D Xfers.
    void *recvXfers[NumLanes] = {nullptr};
    uint64_t recvXferSzs[NumLanes] = {0};
    id = 0;
    uint32_t numRecvXfers = 0;
    for (void *buffer_ptr : buf_ptrs) {
      if (buffer_ptr) {
        emisArgBuf_t *ab = &AB[id];
        // Reset ArgBuf tracker
        emisExtractArgBuf((char *)buffer_ptr, ab);
        for (uint32_t idx = 0; idx < ab->NumSendXfers; idx++)
          emisSkipXferArgSet(ab);
        for (uint32_t idx = 0; idx < ab->NumRecvXfers; idx++) {
          numRecvXfers++;
          void *devAddr = (void *)*((uint64_t *)ab->argptr);
          recvXfers[id] = D2HAddrList[devAddr];
          recvXferSzs[id] =
              (((uint64_t)*((size_t *)(ab->argptr + sizeof(void *)))) &
               0x00000000FFFFFFFF);
          emisSkipXferArgSet(ab);
        }
        id++;
      }
    }
    if (numRecvXfers)
      port.send_n(recvXfers, recvXferSzs);

    // Cleanup all host allocated transfer buffers
    id = 0;
    for (void *buffer_ptr : buf_ptrs) {
      if (buffer_ptr) {
        emisArgBuf_t *ab = &AB[id];
        // Reset the ArgBuf tracker ab
        emisExtractArgBuf((char *)buffer_ptr, ab);
        // Cleanup host allocated send Xfers
        for (uint32_t idx = 0; idx < ab->NumSendXfers; idx++) {
          void *devAddr = (void *)*((uint64_t *)ab->argptr);
          void *hostAddr = D2HAddrList[devAddr];
          delete[] reinterpret_cast<char *>(hostAddr);
          emisSkipXferArgSet(ab);
        }
        // Cleanup host allocated bufs
        for (uint32_t idx = 0; idx < ab->NumRecvXfers; idx++) {
          void *devAddr = (void *)*((uint64_t *)ab->argptr);
          void *hostAddr = D2HAddrList[devAddr];
          delete[] reinterpret_cast<char *>(hostAddr);
          emisSkipXferArgSet(ab);
        }
        id++;
      }
    }

    port.send([&](::rpc::Buffer *Buffer, uint32_t ID) {
      Buffer->data[0] = static_cast<uint64_t>(Results[ID]);
      delete[] reinterpret_cast<char *>(buf_ptrs[ID]);
    });

    break;
  } // END CASE OFFLOAD_EMISSARY_DM

  default: {
    return ::rpc::RPC_UNHANDLED_OPCODE;
    break;
  }
  }
  return ::rpc::RPC_SUCCESS;
} // end handle_emissary_impl

} // namespace internal
} // namespace LIBC_NAMESPACE_DECL

namespace LIBC_NAMESPACE_DECL {
namespace rpc {
LIBC_INLINE ::rpc::Status handleEmissaryOpcodes(::rpc::Server::Port &port,
                                                uint32_t NumLanes) {
  if (NumLanes == 1)
    return internal::handle_emissary_impl<1>(port);
  else if (NumLanes == 32)
    return internal::handle_emissary_impl<32>(port);
  else if (NumLanes == 64)
    return internal::handle_emissary_impl<64>(port);
  else
    return ::rpc::RPC_ERROR;
}

} // namespace rpc
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_RPC_EMISSARY_RPC_SERVER_H
