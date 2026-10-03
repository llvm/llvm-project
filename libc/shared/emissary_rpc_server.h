//===-- Shared memory RPC server instantiation ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is an extension of rpc_server.h
//
// Consumers must add the Clang resource header directory to the include path
// when compiling translation units that include this header (directly or via
// <shared/emissary_rpc_server.h>). EmissaryIds.h is installed there, not under
// lib/llvm/include:
//
//   -I$("$CXX" -print-resource-dir)/include
//
// Typical HIP/OpenMP demo builds also pass -I for lib/llvm/include (or
// llvm/include) so that <shared/emissary_rpc_server.h> resolves.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_RPC_EMISSARY_RPC_SERVER_H
#define LLVM_LIBC_SRC___SUPPORT_RPC_EMISSARY_RPC_SERVER_H

#if __has_include("../clang/lib/Headers/EmissaryIds.h")
#include "../clang/lib/Headers/EmissaryIds.h"
#else
#include "EmissaryIds.h"
#endif

#include "rpc.h"
#include "rpc_opcodes.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>

//===----------------------------------------------------------------------===//
// Emissary host handler registry
//
// Runtime registry that maps an Emissary API id to the host handler that
// services it. It lets a client library register its dispatcher at load time
// so the RPC server (EmissaryTop) can route requests without a compile-time
// switch over every known client.
//
// The registry is a fixed-size table with no dynamic allocation.
// EmissaryRegister and EmissaryLookup have C linkage so a client shared object
// can register against the server without C++ name-mangling coupling. The
// backing table is a C++17 inline variable, so all translation units in a
// program share one instance; on ELF it also merges across shared objects
// under default visibility, letting a client .so and the server share the same
// table.
//===----------------------------------------------------------------------===//

/// Upper bound on the number of distinct Emissary API ids. The table is indexed
/// directly by \c emisid, so this also bounds the largest id that can be
/// registered. It comfortably exceeds the current \c offload_emis_id_t range
/// and leaves room for reserved/dynamic ids.
#define EMISSARY_MAX_REGISTERED_IDS 64

/// Host handler for one Emissary API id. The signature matches the per-client
/// dispatchers (\c EmissaryMPI, \c EmissaryHDF5, ...): it receives the RPC data
/// buffer, the decoded argument descriptor, and the unpacked argument vector.
using EmissaryHandler_t = EmissaryReturn_t (*)(char *data, emisArgBuf_t *ab,
                                               emis_argptr_t *args[]);

namespace emissary_registry_detail {
/// Backing table, indexed directly by Emissary API id. As a C++17 inline
/// variable it has exactly one instance across the whole program, but that
/// only holds across shared objects if the symbol keeps default visibility
/// *and* the linker is told to export it. Two things are needed to make that
/// true even when a consuming DSO is linked with an explicit
/// `--version-script` (as libomptarget/liboffload are) and/or
/// -fvisibility-inlines-hidden (LLVM's default project-wide flag), either of
/// which would otherwise localize this vague-linkage symbol into each DSO and
/// silently defeat the cross-DSO sharing this registry depends on:
///  1. Explicit default visibility, so the compiler doesn't hide it.
///  2. A stable `asm` symbol name, so it can be listed by name in a
///     `--version-script` `global:` clause without embedding an
///     Itanium-mangled C++ symbol (`_ZN...`) in linker input -- mangled names
///     are an ABI/compiler-version implementation detail, not something
///     version scripts should hardcode.
/// \internal
__attribute__((visibility("default"))) inline EmissaryHandler_t
    Table[EMISSARY_MAX_REGISTERED_IDS] asm("EmissaryRegistryTable") = {};
} // namespace emissary_registry_detail

extern "C" {

/// Register a host handler for an Emissary API id.
///
/// \param emisid the Emissary API id (an \c offload_emis_id_t value or a
///   reserved dynamic id) to associate with \p Handler.
/// \param Handler the host dispatcher to invoke for \p emisid; must not be
///   null.
/// \returns \c true on success; \c false if \p emisid is out of range,
///   \p Handler is null, or a *different* handler is already registered for
///   \p emisid. Re-registering the identical handler is idempotent and
///   succeeds.
///
/// Explicit default visibility (see \c Table above): this symbol must merge
/// across DSOs even when the caller is built with -fvisibility-inlines-hidden.
__attribute__((visibility("default"))) inline bool
EmissaryRegister(unsigned int emisid, EmissaryHandler_t Handler) {
  if (emisid >= EMISSARY_MAX_REGISTERED_IDS || Handler == nullptr)
    return false;
  EmissaryHandler_t &Slot = emissary_registry_detail::Table[emisid];
  // Reject last-wins: two libraries claiming the same id is a configuration
  // error, not something to silently overwrite.
  if (Slot != nullptr && Slot != Handler)
    return false;
  Slot = Handler;
  return true;
}

/// Look up the host handler registered for an Emissary API id.
///
/// \param emisid the Emissary API id to look up.
/// \returns the registered handler, or null if \p emisid is out of range or no
///   handler is registered for it.
///
/// Explicit default visibility (see \c Table above): this symbol must merge
/// across DSOs even when the caller is built with -fvisibility-inlines-hidden.
__attribute__((visibility("default"))) inline EmissaryHandler_t
EmissaryLookup(unsigned int emisid) {
  if (emisid >= EMISSARY_MAX_REGISTERED_IDS)
    return nullptr;
  return emissary_registry_detail::Table[emisid];
}

} // extern "C"

// No Emissary API host handler is declared or called here by name.
// Every client -- MPI, HDF5, PRINT, and RESERVE -- self-registers its host
// dispatcher with the runtime registry defined above, so EmissaryTop routes all
// of them through EmissaryLookup without a compile-time weak symbol or switch
// case. PRINT (device printf/fprintf) is no longer built into this header
// either: it is an ordinary client library (libemissary_print) that an
// application links like libemissary_mpi. RESERVE is registered by the
// user/reserved client library.
extern "C" {
/// Optional FORCE_OPT=1 SDMA path for device MPI Put/Get (libemissary_mpi).
/// This is an internal optimization hook, not an Emissary API, so it keeps its
/// weak-stub design: libLLVMOffload links without libemissary_mpi; the app
/// overrides with a strong definition from libemissary_mpi when FORCE_OPT SDMA
/// is used.
__attribute__((weak)) int
emissary_mpi_sdma_try_dm_buffer(char *rpc_buffer,
                                unsigned long long *out_result) {
  (void)rpc_buffer;
  (void)out_result;
  return -1;
}
} // end extern "C"

namespace rpc {
namespace internal {

// emisExtractArgBuf extract ArgBuf using protocol EmitEmissaryExec makes.
static void emisExtractArgBuf(char *data, emisArgBuf_t *ab) {

  uint32_t *Int32Data = (uint32_t *)data;
  ab->DataLen = Int32Data[0];
  ab->NumArgs = Int32Data[1];

  // Note: while the data buffer contains all args including strings,
  // ab->DataLen does not include strings. It only counts header, keys,
  // and aligned numerics.

  ab->keyptr = data + (2 * sizeof(int));
  ab->argptr = ab->keyptr + (ab->NumArgs * sizeof(int));
  ab->strptr = data + (size_t)ab->DataLen;
  int AlignFill = 0;
  if (((size_t)ab->argptr) % (size_t)8) {
    ab->argptr += 4;
    AlignFill = 4;
  }

  // Extract the two emissary identifiers and number of send
  // and recv device data transfers. These are 4 16 bit values
  // packed into a single 64-bit field.
  uint64_t Arg1 = *(uint64_t *)ab->argptr;
  ab->emisid = (unsigned int)((Arg1 >> 48) & 0xFFFF);
  ab->emisfnid = (unsigned int)((Arg1 >> 32) & 0xFFFF);
  ab->NumSendXfers = (unsigned int)((Arg1 >> 16) & 0xFFFF);
  ab->NumRecvXfers = (unsigned int)((Arg1) & 0xFFFF);

  // skip the uint64_t emissary id arg which is first arg in _emissary_exec.
  ab->keyptr += sizeof(int);
  ab->argptr += sizeof(uint64_t);
  ab->NumArgs -= 1;

  // data_not_used used for testing consistency.
  ab->data_not_used =
      (size_t)(ab->DataLen) - (((size_t)(3 + ab->NumArgs) * sizeof(int)) +
                               sizeof(uint64_t) + AlignFill);

  // Ensure first arg after emissary id arg is aligned.
  if (((size_t)ab->argptr) % (size_t)8) {
    ab->argptr += 4;
    ab->data_not_used -= 4;
  }
}

/// Get uint32 value extended to uint64_t value from a char ptr
static uint64_t getUInt32(char *val) {
  uint32_t i32 = *(uint32_t *)val;
  return (uint64_t)i32;
}

/// Get uint64_t value from a char ptr
static uint64_t getUInt64(char *val) { return *(uint64_t *)val; }

// build argument array to create call to variadic wrappers
static uint32_t
emissaryBuildVargs(int NumArgs, char *keyptr, char *dataptr, char *strptr,
                   unsigned long long *data_not_used, emis_argptr_t *a[],
                   std::unordered_map<void *, void *> *D2HAddrList) {
  size_t NumBytes;
  size_t BytesConsumed;
  size_t StrSz;
  size_t FillerNeeded;
  uint32_t ArgCount = 0;
  for (int ArgNum = 0; ArgNum < NumArgs; ArgNum++) {
    NumBytes = 0;
    StrSz = 0;
    unsigned int Key = *(unsigned int *)keyptr;
    unsigned int EmisId = Key >> 16;
    unsigned int NumBits = (Key << 16) >> 16;

    switch (EmisId) {
    case EmisFloatTy:
      NumBytes = NumBits / 8;
      BytesConsumed = NumBytes;
      FillerNeeded = ((size_t)dataptr) % NumBytes;
      if (FillerNeeded) {
        dataptr += FillerNeeded;
        BytesConsumed += FillerNeeded;
      }
      if ((*data_not_used) < BytesConsumed)
        return _ERC_DATA_USED_ERROR;

      if (NumBytes == 4)
        a[ArgCount] = (emis_argptr_t *)getUInt32(dataptr);
      else {
        double *value = (double *)dataptr;
        a[ArgCount] = (emis_argptr_t *)(uint64_t)*value;
      }
      break;

    case EmisIntegerTy:
      NumBytes = NumBits / 8;
      BytesConsumed = NumBytes;
      FillerNeeded = ((size_t)dataptr) % NumBytes;
      if (FillerNeeded) {
        dataptr += FillerNeeded;
        BytesConsumed += FillerNeeded;
      }
      if ((*data_not_used) < BytesConsumed)
        return _ERC_DATA_USED_ERROR;

      if (NumBytes == 4)
        a[ArgCount] = (emis_argptr_t *)getUInt32(dataptr);
      else
        a[ArgCount] = (emis_argptr_t *)getUInt64(dataptr);
      break;

    case EmisPointerTy: {
      if (NumBits == 1) { // This is a pointer to string
        NumBytes = 4;
        BytesConsumed = NumBytes;
        StrSz = (size_t)*(unsigned int *)dataptr;
        if ((*data_not_used) < BytesConsumed)
          return _ERC_DATA_USED_ERROR;
        a[ArgCount] = (emis_argptr_t *)((char *)strptr);
      } else {
        NumBytes = 8;
        BytesConsumed = NumBytes;
        FillerNeeded = ((size_t)dataptr) % NumBytes;
        if (FillerNeeded) {
          dataptr += FillerNeeded; // dataptr is now aligned
          BytesConsumed += FillerNeeded;
        }
        if ((*data_not_used) < BytesConsumed)
          return _ERC_DATA_USED_ERROR;
        a[ArgCount] = (emis_argptr_t *)getUInt64(dataptr);
      }
      if (D2HAddrList) {
        auto Found = D2HAddrList->find((void *)a[ArgCount]);
        if (Found != D2HAddrList->end())
          a[ArgCount] = (emis_argptr_t *)Found->second;
      }
    } break;

    default:
      return _ERC_INVALID_ID_ERROR;
    }
    // Move to next argument
    dataptr += NumBytes;
    strptr += StrSz;
    *data_not_used -= BytesConsumed;
    keyptr += 4;
    ArgCount++;
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
  // Registry-only dispatch (D1): every Emissary API -- MPI, HDF5, PRINT,
  // RESERVE, and any out-of-tree client -- is serviced through the runtime
  // registry. A client's handler is present because its library
  // self-registered at load time. There is no per-client switch and no weak
  // symbol fallback: an unregistered id (for example the reserved-but-unused
  // EMIS_ID_FORTRT) is simply unsupported.
  if (ab->emisid == EMIS_ID_INVALID) {
    fprintf(stderr, "Emissary (host execution) got invalid EMIS_ID\n");
    return (EmissaryReturn_t)0;
  }

  EmissaryHandler_t Handler = EmissaryLookup(ab->emisid);
  if (Handler == nullptr) {
    fprintf(stderr,
            "Emissary (host execution) EMIS_ID:%d fnid:%d not supported\n",
            ab->emisid, ab->emisfnid);
    return (EmissaryReturn_t)0;
  }

  emis_argptr_t **args = (emis_argptr_t **)aligned_alloc(
      sizeof(emis_argptr_t), ab->NumArgs * sizeof(emis_argptr_t *));

  // Build the unpacked argument vector against a scratch copy of data_not_used
  // so the buffer descriptor (ab) is left pristine for the handler. PRINT walks
  // the raw buffer itself and relies on ab->data_not_used being intact; the
  // other handlers use only the argument vector, so this is safe for all of
  // them and keeps a single uniform dispatch path.
  unsigned long long data_not_used = ab->data_not_used;
  if (emissaryBuildVargs(ab->NumArgs, ab->keyptr, ab->argptr, ab->strptr,
                         &data_not_used, &args[0],
                         D2HAddrList) != _ERC_SUCCESS) {
    free(args);
    return (EmissaryReturn_t)0;
  }

  EmissaryReturn_t result = Handler(data, ab, args);
  free(args);
  return result;
}

// -----------------------------------------------------------------
// -- Handle OFFLOAD_EMISSARY and OFFLOAD_EMISSARY_DM opcodes     --
// -- handleEmissaryImpl calls EmissaryTop for each active lane --
// -----------------------------------------------------------------
template <uint32_t NumLanes>
inline RPCStatus handleEmissaryImpl(Server::Port &port) {

  switch (port.get_opcode()) {

  // This case handles the device function __llvm_emissary_rpc for emissary
  // APIs that require no d2h or h2d memory transfer.
  case OFFLOAD_EMISSARY: {
    uint64_t Sizes[NumLanes] = {0};
    unsigned long long Results[NumLanes] = {0};
    void *BufPtrs[NumLanes] = {nullptr};
    port.recv_n(BufPtrs, Sizes, [&](uint64_t Size) { return new char[Size]; });
    uint32_t id = 0;
    for (void *BufferPtr : BufPtrs) {
      if (BufferPtr) {
        emisArgBuf_t ab;
        emisExtractArgBuf((char *)BufferPtr, &ab);
        Results[id++] = EmissaryTop((char *)BufferPtr, &ab, nullptr);
      }
    }
    port.send([&](::rpc::Buffer *Buffer, uint32_t ID) {
      Buffer->data[0] = static_cast<uint64_t>(Results[ID]);
    });
    for (void *BufferPtr : BufPtrs) {
      if (BufferPtr) {
        delete[] reinterpret_cast<char *>(BufferPtr);
      }
    }
    break;
  }

  // This case handles the device function __llvm_emissary_rpc_dm for emissary
  // APIs require D2H or H2D transfer vectors to be processed through the port.
  // FIXME: test with multiple transfer vectors of the same type.
  case OFFLOAD_EMISSARY_DM: {
    uint64_t Sizes[NumLanes] = {0};
    unsigned long long Results[NumLanes] = {0};
    void *BufPtrs[NumLanes] = {nullptr};
    port.recv_n(BufPtrs, Sizes, [&](uint64_t Size) { return new char[Size]; });

    uint32_t id = 0;
    emisArgBuf_t AB[NumLanes];
    std::unordered_map<void *, void *> D2HAddrList;
    void *Xfers[NumLanes] = {nullptr};
    void *DevXfers[NumLanes] = {nullptr};
    uint64_t XferSzs[NumLanes] = {0};
    bool SdmaHandled[NumLanes] = {false};
    uint32_t TotalSendXfers = 0;
    id = 0;

    for (void *BufferPtr : BufPtrs) {
      if (BufferPtr) {

        emisArgBuf_t *ab = &AB[id];
        emisExtractArgBuf((char *)BufferPtr, ab);
        unsigned long long SdmaResult = 0;
        if (emissary_mpi_sdma_try_dm_buffer((char *)BufferPtr, &SdmaResult) ==
            0) {
          Results[id] = SdmaResult;
          SdmaHandled[id] = true;
          id++;
          continue;
        }
        for (uint32_t idx = 0; idx < ab->NumSendXfers; idx++) {
          TotalSendXfers++;
          DevXfers[id] = (void *)*((uint64_t *)ab->argptr);
          XferSzs[id] = (size_t)*((size_t *)(ab->argptr + sizeof(void *)));
          emisSkipXferArgSet(ab);
        }
        // Allocate the host space for the receive Xfers
        for (uint32_t idx = 0; idx < ab->NumRecvXfers; idx++) {
          void *DevAddr = (void *)*((uint64_t *)ab->argptr);
          size_t DevSz = (((size_t)*((size_t *)(ab->argptr + sizeof(void *)))) &
                          0x00000000FFFFFFFF);
          void *HostAddr = new char[DevSz];
          D2HAddrList.insert(std::pair<void *, void *>(DevAddr, HostAddr));
          emisSkipXferArgSet(ab);
        }
        id++;
      }
    }

    // recv_n for device send_n into new host-allocated Xfers
    if (TotalSendXfers)
      port.recv_n(Xfers, XferSzs,
                  [&](uint64_t Size) { return new char[Size]; });

    // Xfers now contains just allocated host addrs for sends and
    // DevXfers contains corresponding DevAddr for those sends
    // Build map to pass to Emissary
    id = 0;
    for (void *Xfer : Xfers) {
      if (Xfer) {
        D2HAddrList.insert(std::pair<void *, void *>(DevXfers[id], Xfer));
        id++;
      }
    }

    // Call EmissaryTop for each active lane
    id = 0;
    for (void *BufferPtr : BufPtrs) {
      if (BufferPtr) {
        if (SdmaHandled[id]) {
          id++;
          continue;
        }
        emisArgBuf_t *ab = &AB[id];
        emisExtractArgBuf((char *)BufferPtr, ab);
        for (uint32_t idx = 0; idx < ab->NumSendXfers; idx++)
          emisSkipXferArgSet(ab);
        for (uint32_t idx = 0; idx < ab->NumRecvXfers; idx++)
          emisSkipXferArgSet(ab);
        Results[id] = EmissaryTop((char *)BufferPtr, ab, &D2HAddrList);
        id++;
      }
    }

    // Process send_n for the H2D Xfers.
    void *recvXfers[NumLanes] = {nullptr};
    uint64_t recvXferSzs[NumLanes] = {0};
    id = 0;
    uint32_t TotalRecvXfers = 0;
    for (void *BufferPtr : BufPtrs) {
      if (BufferPtr) {
        if (SdmaHandled[id]) {
          id++;
          continue;
        }
        emisArgBuf_t *ab = &AB[id];
        // Reset ArgBuf tracker
        emisExtractArgBuf((char *)BufferPtr, ab);
        for (uint32_t idx = 0; idx < ab->NumSendXfers; idx++)
          emisSkipXferArgSet(ab);
        for (uint32_t idx = 0; idx < ab->NumRecvXfers; idx++) {
          TotalRecvXfers++;
          void *DevAddr = (void *)*((uint64_t *)ab->argptr);
          recvXfers[id] = D2HAddrList[DevAddr];
          recvXferSzs[id] =
              (((uint64_t)*((size_t *)(ab->argptr + sizeof(void *)))) &
               0x00000000FFFFFFFF);
          emisSkipXferArgSet(ab);
        }
        id++;
      }
    }
    if (TotalRecvXfers)
      port.send_n(recvXfers, recvXferSzs);

    // Cleanup all host allocated transfer buffers
    id = 0;
    for (void *BufferPtr : BufPtrs) {
      if (BufferPtr) {
        if (SdmaHandled[id]) {
          id++;
          continue;
        }
        emisArgBuf_t *ab = &AB[id];
        // Reset the ArgBuf tracker ab
        emisExtractArgBuf((char *)BufferPtr, ab);
        // Cleanup host allocated send Xfers
        for (uint32_t idx = 0; idx < ab->NumSendXfers; idx++) {
          void *DevAddr = (void *)*((uint64_t *)ab->argptr);
          void *HostAddr = D2HAddrList[DevAddr];
          delete[] reinterpret_cast<char *>(HostAddr);
          emisSkipXferArgSet(ab);
        }
        // Cleanup host allocated bufs
        for (uint32_t idx = 0; idx < ab->NumRecvXfers; idx++) {
          void *DevAddr = (void *)*((uint64_t *)ab->argptr);
          void *HostAddr = D2HAddrList[DevAddr];
          delete[] reinterpret_cast<char *>(HostAddr);
          emisSkipXferArgSet(ab);
        }
        id++;
      }
    }

    port.send([&](::rpc::Buffer *Buffer, uint32_t ID) {
      Buffer->data[0] = static_cast<uint64_t>(Results[ID]);
      delete[] reinterpret_cast<char *>(BufPtrs[ID]);
    });

    break;
  } // END CASE OFFLOAD_EMISSARY_DM

  default: {
    return ::rpc::RPC_UNHANDLED_OPCODE;
    break;
  }
  }
  return ::rpc::RPC_SUCCESS;
} // end handleEmissaryImpl

} // namespace internal

// Handles any opcode generated from emissary client code.
inline RPCStatus handleEmissaryOpcodes(Server::Port &port, uint32_t num_lanes) {
  switch (num_lanes) {
  case 1:
    return internal::handleEmissaryImpl<1>(port);
  case 32:
    return internal::handleEmissaryImpl<32>(port);
  case 64:
    return internal::handleEmissaryImpl<64>(port);
  default:
    return RPC_ERROR;
  }
}

} // namespace rpc

#endif // LLVM_LIBC_SRC___SUPPORT_RPC_EMISSARY_RPC_SERVER_H
