//===----------------  openmp/device/include/EmissaryMPI.h  ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// EmissaryMPI.h This include must be included by MPI application
//
//===----------------------------------------------------------------------===//
#include "EmissaryIds.h"
#include <stdarg.h>
#include <unordered_map>

typedef enum {
  _MPI_INVALID,
  _MPI_Send_idx,
  _MPI_Recv_idx,
  _MPI_Allreduce_idx,
  _MPI_Reduce_idx,
} offload_emis_mpi_t;

// -------- DELETE THIS BLOCK WHEN MPI_Type_size on GPU WORKS ----------------
// Emissary_Initialize_MPI builds a table of lengths for each each MPI_Datatype.
// That table is called _mpi_type_lens and passed to the device.
// We need this because we do not yet have a GPU version of MPI_Type_size.
// If we did we can avoid the table search for datatype_size. This is
// how MPI datatype length should be calculated.
//   int datatype_size;
//   MPI_Type_size(v2,&datatype_size) ;
// Delete this table search when we have a working MPI_Type_size.
#define _MPI_DATATYPES 5
typedef struct mpi_type_len_t {
  uint64_t dt_signature;
  uint32_t dt_size;
} mpi_type_len_t;
#pragma omp begin declare target
mpi_type_len_t _mpi_type_lens[_MPI_DATATYPES];
#pragma omp end declare target
void Emissary_Initialize_MPI() {
  MPI_Datatype _mpi_int = MPI_INT;
  MPI_Datatype _mpi_float = MPI_FLOAT;
  MPI_Datatype _mpi_unsigned = MPI_UNSIGNED;
  MPI_Datatype _mpi_double = MPI_DOUBLE;
  MPI_Datatype _mpi_char = MPI_CHAR;
  _mpi_type_lens[0] = {(uint64_t)_mpi_int, 4};
  _mpi_type_lens[1] = {(uint64_t)_mpi_unsigned, 4};
  _mpi_type_lens[2] = {(uint64_t)_mpi_float, 4};
  _mpi_type_lens[3] = {(uint64_t)_mpi_double, 8};
  _mpi_type_lens[4] = {(uint64_t)_mpi_char, 1};
#pragma omp target update to(_mpi_type_lens[0 : _MPI_DATATYPES])
}
// -------- END BLOCK TO DELETE WHEN MPI_Type_size on GPU WORKS ---------------

///  Device stubs must use the identical host API interface.
///  Stubs call _emissary_exec with additional args that include
///  the identifier and additional D2H and H2D transfer vectors.
///  whose params include an identifier
///
#if defined(__NVPTX__) || defined(__AMDGCN__)

extern "C" int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
                        int dest, int tag, MPI_Comm comm) {
  uint64_t dt_signature = (uint64_t)datatype;
  int datatype_size = 8; // Default in case we do not have it in our table.
  for (int i = 0; i < _MPI_DATATYPES; i++)
    if (_mpi_type_lens[i].dt_signature == dt_signature) {
      datatype_size = _mpi_type_lens[i].dt_size;
      break;
    }
  return (int)_emissary_exec(
      // The emissary identifier is a static 64 bit field that encodes
      // the emissary id, emissary function, D2H Xfer cnt, and H2D Xfer cnt.
      _PACK_EMIS_IDS(EMIS_ID_MPI, _MPI_Send_idx, 1, 0),
      // Each D2H transfer vector needs a pair of args to describe the xfer
      // The first is the device pointer, the 2nd is size.
      buf, (int)count * datatype_size,
      // These are the actual 6 MPI_Send Args passed directly from the params
      buf, count, datatype, dest, tag, comm);
}
extern "C" int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source,
                        int tag, MPI_Comm comm, MPI_Status *st) {
  uint64_t dt_signature = (uint64_t)datatype;
  int datatype_size = 8;
  for (int i = 0; i < _MPI_DATATYPES; i++)
    if (_mpi_type_lens[i].dt_signature == dt_signature) {
      datatype_size = _mpi_type_lens[i].dt_size;
      break;
    }
  return (int)_emissary_exec(_PACK_EMIS_IDS(EMIS_ID_MPI, _MPI_Recv_idx, 0, 1),
                             buf,
                             (int)count * datatype_size, // This is a H2D Xfer
                             buf, count, datatype, source, tag, comm, st);
}
extern "C" int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                             MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  return (int)_emissary_exec(
      _PACK_EMIS_IDS(EMIS_ID_MPI, _MPI_Allreduce_idx, 1, 1), sendbuf, recvbuf,
      count, datatype, op, comm);
}
extern "C" int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
                          MPI_Datatype datatype, MPI_Op op, int root,
                          MPI_Comm comm) {
  return (int)_emissary_exec(_PACK_EMIS_IDS(EMIS_ID_MPI, _MPI_Reduce_idx, 1, 1),
                             sendbuf, recvbuf, count, datatype, op, root, comm);
}

#else

/// Host variadic wrapper functions.
extern "C" {
extern int V_MPI_Send(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *v0 = va_arg(args, void *);
  int v1 = va_arg(args, int);
  MPI_Datatype v2 = va_arg(args, MPI_Datatype);
  int v3 = va_arg(args, int);
  int v4 = va_arg(args, int);
  MPI_Comm v5 = va_arg(args, MPI_Comm);
  va_end(args);
  int rval = MPI_Send(v0, v1, v2, v3, v4, v5);
  return rval;
}
extern int V_MPI_Recv(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *v0 = va_arg(args, void *);
  int v1 = va_arg(args, int);
  MPI_Datatype v2 = va_arg(args, MPI_Datatype);
  int v3 = va_arg(args, int);
  int v4 = va_arg(args, int);
  MPI_Comm v5 = va_arg(args, MPI_Comm);
  MPI_Status *v6 = va_arg(args, MPI_Status *);
  va_end(args);
  int rval = MPI_Recv(v0, v1, v2, v3, v4, v5, v6);
  return rval;
}
extern int V_MPI_Allreduce(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *buf = va_arg(args, void *);
  void *recvbuf = va_arg(args, void *);
  int count = va_arg(args, int);
  MPI_Datatype datatype = va_arg(args, MPI_Datatype);
  MPI_Op op = va_arg(args, MPI_Op);
  MPI_Comm comm = va_arg(args, MPI_Comm);
  va_end(args);
  int rval = MPI_Allreduce(buf, recvbuf, count, datatype, op, comm);
  return rval;
}
extern int V_MPI_Reduce(void *fnptr, ...) {
  va_list args;
  va_start(args, fnptr);
  void *buf = va_arg(args, void *);
  void *recvbuf = va_arg(args, void *);
  int count = va_arg(args, int);
  MPI_Datatype datatype = va_arg(args, MPI_Datatype);
  MPI_Op op = va_arg(args, MPI_Op);
  int root = va_arg(args, int);
  MPI_Comm comm = va_arg(args, MPI_Comm);
  va_end(args);
  int rval = MPI_Reduce(buf, recvbuf, count, datatype, op, root, comm);
  return rval;
}

/// EmissaryMPI function selector
EmissaryReturn_t EmissaryMPI(char *data, emisArgBuf_t *ab, emis_argptr_t *a[]) {

  switch (ab->emisfnid) {
  case _MPI_Send_idx: {
    void *fnptr = (void *)V_MPI_Send;
    int return_value_int =
        V_MPI_Send(fnptr, a[0], a[1], a[2], a[3], a[4], a[5]);
    return (EmissaryReturn_t)return_value_int;
  }
  case _MPI_Recv_idx: {
    void *fnptr = (void *)V_MPI_Recv;
    int return_value_int =
        V_MPI_Recv(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    return (EmissaryReturn_t)return_value_int;
  }
  case _MPI_Allreduce_idx: {
    void *fnptr = (void *)V_MPI_Allreduce;
    int return_value_int =
        V_MPI_Allreduce(fnptr, a[0], a[1], a[2], a[3], a[4], a[5]);
    return (EmissaryReturn_t)return_value_int;
  }
  case _MPI_Reduce_idx: {
    void *fnptr = (void *)V_MPI_Reduce;
    int return_value_int =
        V_MPI_Reduce(fnptr, a[0], a[1], a[2], a[3], a[4], a[5], a[6]);
    return (EmissaryReturn_t)return_value_int;
  }
  }
  return (EmissaryReturn_t)0;
}

} // end extern "C"

#endif
