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
/// Called by EmissaryTop to support printf/fprintf/asan report externally
__attribute((weak)) EmissaryReturn_t EmissaryPrint(char *data, emisArgBuf_t *ab,
                                                   emis_argptr_t *arg[]);
} // end extern "C"
} // namespace EmissaryExternal

namespace LIBC_NAMESPACE_DECL {
namespace internal {

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
    unsigned int emis_id = key >> 16;
    unsigned int numbits = (key << 16) >> 16;

    switch (emis_id) {
    case EmisFloatTy:
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

    case EmisIntegerTy:
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

    case EmisPointerTy: {
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
    result = EmissaryExternal::EmissaryPrint(data, ab, args);
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
