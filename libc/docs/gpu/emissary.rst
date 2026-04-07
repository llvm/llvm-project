.. _libc_gpu_emissary:

=============
Emissary APIs
=============

Emissary APIs
-------------

The libc GPU feature of LLVM offload provides the ability to execute 
libc functions on the host initiated from GPU code including OpenMP
target regions, CUDA kernels, or HIP kernels. 
This libc-gpu capability uses the LLVM offload RPC mechanism. 
An extension to libc-gpu is the abiity to execute arbitrary functions
initiated by GPU source code. This extension is called ``Emissary APIs``. 

Emissary APIs allow host API maintainers to easily maintain and distribute 
a platform-specific Emissary API directly callable from the GPU. 
The LLVM runtime does not link to the platform-specific host runtime until
th application links to the host runtime. For standarized APIs, such as MPI,
emissary APIs allows multiple platform-specific implementations of MPI.
This results in increased application portability. But the primary 
benefit of Emissary APIs is the ability to execute host functions without
terminating the GPU kernel or code.

An Emissary API implementation consists of a header file and a simple c++ 
source file. The same implementation can be used for OpenMP, HIP, or Cuda.
The c++ source file can be compiled as part of the host API library.

This architecture allows host API maintainers to maintain the Emissary API
externally.  That is, no ,change to the compiler or runtime is required.
Maintainers can implement any subset of the host API in their emissary
API implementation. Users attempting to use an unimplemented function
from GPU code would get the same unresolved GPU reference they get
without an Emissary API implementation. 

Because they execute the actual external host functions, the server
implementation cannot be directly linked to the LLVM runtime. The LLVM
runtime in the emissary support provides a weak external reference
to a single master function for the Emissary API. The external API
maintainer provides this master function consisting of a case clause and
wrapper function for each implemented function. There are a number of ways
the API maintainer can package and distribute emissary support for a
platform-specific API. Compiling the master function into the host library
and distributing a device header file is typical.

This external implementation architecture provides the ability to have 
different platform-specific APIs for standard interface libraries
such as ROCm MPI or CUDA MPI. 

In this documentation, we provide an MPI example with a few MPI 
functions as a demonstration. 

EmisssaryMPI Example
--------------------

External Emissary APIs require an external library such as OpenMPI. 
This example source shows the execution of MPI_Send and MPI_Recv
from an OpenMP target region. 


.. code-block:: c++

  // 
  //  EmissaryMPI_example.cpp
  //
  #include <EmissaryMPI.h>
  #include <mpi.h>
  #include <omp.h>
  #include <stdio.h>
  #define VSIZE 5000
  int main(int argc, char *argv[]) {
    int numranks, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm _mpi_comm = MPI_COMM_WORLD;
    MPI_Datatype _mpi_int = MPI_INT;
    int rc = 0;
    int *send_recv_buffer = (int *)malloc(VSIZE * sizeof(int));
    printf("R%d: Number of Ranks= %d ArraySize:%d\n", rank, numranks, VSIZE);
    #pragma omp target teams distribute parallel for map(tofrom : rc)          \
      map(to : send_recv_buffer[0 : VSIZE])
    for (int i = 0; i < VSIZE; i++) {
      if (rank == 0) {
        send_recv_buffer[i] = -i;
        MPI_Send(&send_recv_buffer[i], 1, _mpi_int, 1, i, _mpi_comm);
      } else {
        MPI_Recv(&send_recv_buffer[i], 1, _mpi_int, 0, i, _mpi_comm,
                 MPI_STATUS_IGNORE);
        if (send_recv_buffer[i] != -i)
          rc = 1;  // FLAG AS ERROR IF NOT EXPECTED.
      }
    }
    MPI_Finalize();
    printf("R%d: === rc === %d\n", rank, rc);
    return rc;
  }

It is worth noting that without Emissary API for MPI, the device link 
step for the above code would fail with unresolved references to 
MPI_Send and MPI_Recv. The include of EmissaryMPI.h resolves this. 

.. code-block:: c++

  //===--------------------------------------------------------------------===//
  // 
  // EmissaryMPI.h : Example device header for EmissaryMPI
  //
  //===--------------------------------------------------------------------===//
  #ifndef OFFLOAD_EMISSARY_MPI_H
  #define OFFLOAD_EMISSARY_MPI_H
  #include "EmissaryIds.h"
  #include <mpi.h>
  #include <stdarg.h>
  typedef enum {
    _MPI_INVALID,
    _MPI_Send_idx,
    _MPI_Recv_idx,
  } offload_emis_mpi_t;
  #if defined(__NVPTX__) || defined(__AMDGCN__)
  //  EmissaryIds.h sets __DEVATTR__ to __device__ when compiling for either
  //  CUDA or HIP. That attribute is not used for OpenMP device compilation.
  __DEVATTR__
  extern "C" int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
                          int dest, int tag, MPI_Comm comm) {
    return (int) _emissary_exec(
        _PACK_EMIS_IDS(EMIS_ID_MPI, _MPI_Send_idx, 1, 0),
        buf, (int)count * 4, buf, count, datatype, dest, tag, comm);
  }
  __DEVATTR__
  extern "C" int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source,
                          int tag, MPI_Comm comm, MPI_Status *st) {
    return (int) _emissary_exec(
        _PACK_EMIS_IDS(EMIS_ID_MPI, _MPI_Recv_idx, 0, 1),
       buf, (int)count * 4, buf, count, datatype, source, tag, comm, st);
  }
  #endif
  #endif // end #ifndef OFFLOAD_EMISSARY_MPI_H

The above is a minimal Emissary API implementation to provide MPI_Send and MPI_Recv
functionality for demonstration.  The host service side to support this is the following:

.. code-block:: c++

  //===--------------------------------------------------------------------===//
  //
  // EmissaryMPI.cpp : Defines the EmissaryMPI master function and variadic
  //                   wrappers for each implemented function. In this example
  //                   only MPI_Send and MPI_Recv are callable by GPU.  
  //
  //===--------------------------------------------------------------------===//
  #include "EmissaryMPI.h"
  #include <EmissaryIds.h>
  #include <mpi.h>
  namespace EmissaryExternal {
  // EmissaryMPI function selector
  extern "C" EmissaryReturn_t EmissaryMPI(char *data, emisArgBuf_t *ab,
                                        emis_argptr_t *a[]) {
    switch (ab->emisfnid) {
    case _MPI_Send_idx: {
      return (EmissaryReturn_t) MPI_Send(
        (const void*)  ((unsigned long long int) a[0]),
        (int)          ((unsigned long long int) a[1]),
        (MPI_Datatype) ((unsigned long long int) a[2]),
        (int)          ((unsigned long long int) a[3]),
        (int)          ((unsigned long long int) a[4]),
        (MPI_Comm)     ((unsigned long long int) a[5]));
    }
    case _MPI_Recv_idx: {
    return (EmissaryReturn_t) MPI_Recv(
      (void*)        ((unsigned long long int) a[0]),
      (int)          ((unsigned long long int) a[1]),
      (MPI_Datatype) ((unsigned long long int) a[2]),
      (int)          ((unsigned long long int) a[3]),
      (int)          ((unsigned long long int) a[4]),
      (MPI_Comm)     ((unsigned long long int) a[5]),
      (MPI_Status *) ((unsigned long long int) a[6]));
    }
    }
    return (EmissaryReturn_t)0;
  }
  }

The above OpenMP user source code can be compiled and executed with
the following shell script:

.. code-block:: sh

  #/bin/bash
  #
  #  demo_mpi.sh
  #
  MPI=${MPI:-~/local/openmpi}
  LLVM_INSTALL=${LLVM_INSTALL:-/work/grodgers/rocm/trunk}
  OFFLOAD_ARCH=${OFFLOAD_ARCH:-gfx90a}
  export PATH=$MPI/bin:$PATH
  [ ! -d "$MPI" ] && echo "MPI:$MPI not found" && exit
  [ ! -d "$LLVM_INSTALL" ] && echo "LLVM_INSTALL:$LLVM_INSTALL not found" && exit
  echo "===1===>  Compiling Host Master Function EmissaryMPI found in EmissaryMPI.cpp"
  $LLVM_INSTALL/bin/clang++ ../EmissaryMPI.cpp  -I.. -I$MPI/include -O3 -c -fPIC -o EmissaryMPI.o
  echo "===2===>  Compiling and linking OpenMP application"
  export OMPI_CC=$LLVM_INSTALL/bin/clang++
  mpic++ -fopenmp --offload-arch=$OFFLOAD_ARCH EmissaryMPI_example.cpp -I.. -Xlinker EmissaryMPI.o
  echo "===3===>  Executing 2 MPI ranks with ./a.out on GPU $OFFLOAD_ARCH"
  mpirun -np 2 a.out 

The shell compiles the host master function EmissaryMPI to resolve the weak 
reference to EmissaryMPI(...) provided by the OpenMP runtime.
which is then linked to the application a.out.  The platform-specific MPI
might put EmissaryMPI in their library.

A typical installation of OpenMPI would not accept device pointers.  However, a platform-specific
implementation might have GPU-aware host library that does recognize device pointers.

To make the typical installation work, EmissaryMPIs provide the definition of 
device-to-host (send) and host-to-device(receive) transfer vectors. 

The number of send transfer
vectors and receive transfer vectors are embedded into the first argument to _emissary_exec
with the macro _PACK_EMIS_IDS. The macro _PACK_EMIS_IDS has 4 16-bit fields: The enum ID 
of the Emissary ID , the function index within that Emissary ID, the number of send 
transfer vectors, and the number of receive transfer vectors. In the above example, 
these first arguments are used in EmissaryMPI.h.
      
.. code-block::

      PACK_EMIS_IDS(EMIS_ID_MPI, _MPI_Send_idx, 1, 0)
      PACK_EMIS_IDS(EMIS_ID_MPI, _MPI_Recv_idx, 0, 1)

Each transfer vector adds two arguments to the call to _emissary_exec: the device pointer 
argument, followed by the length in number of bytes to transfer. Obviously the use of
transfer vectors slows runtime execution to allocate and move data.

If the host platform-specific library was GPU-aware, no transfer vectors would be required.
In this case the first arg in the header to _emissary_exec would be:

.. code-block::

      PACK_EMIS_IDS(EMIS_ID_MPI, _MPI_Send_idx, 0, 0)
      PACK_EMIS_IDS(EMIS_ID_MPI, _MPI_Recv_idx, 0, 0)

This would be significantly faster than the typical OpenMPI that require
transfer vectors.  

PACK_EMIS_IDS generates a compile time 64-bit constant to the first argument
to _emissary_exec, followed by transfer vectors (if any) and then followed by
the actual arguments to the host function. The device pass of the clang compiler
emits code to pack the arguments into a buffer and generates a call to the
proper RPC function.

The use of Emissary API relieves the API maintainer from implementing different
RPC functions to manage different sets of arguments. The later approach is what
is done in the implementation of most libc functions. Emissary APIs are 
useful when there are complex sets of arguments such as in IO APIs. 
