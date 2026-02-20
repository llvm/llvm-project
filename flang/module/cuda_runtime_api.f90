!===-- module/cuda_runtime_api.f90 -----------------------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===------------------------------------------------------------------------===!

module cuda_runtime_api
implicit none

integer, parameter :: cuda_stream_kind = int_ptr_kind()

interface cudaforgetdefaultstream
  integer(kind=cuda_stream_kind) function cudagetstreamdefaultarg(devptr)
    import cuda_stream_kind
    !DIR$ IGNORE_TKR (TKR) devptr
    integer, device  :: devptr(*)
  end function
  integer(kind=cuda_stream_kind) function cudagetstreamdefaultnull()
    import cuda_stream_kind
  end function
end interface

interface cudaforsetdefaultstream
  integer function cudasetstreamdefault(stream)
    import cuda_stream_kind
    !DIR$ IGNORE_TKR (K) stream
    integer(kind=cuda_stream_kind), value :: stream
  end function
  integer function cudasetstreamarray(devptr, stream)
    import cuda_stream_kind
    !DIR$ IGNORE_TKR (K) stream, (TKR) devptr
    integer, device  :: devptr(*)
    integer(kind=cuda_stream_kind), value :: stream
  end function
end interface

interface cudaStreamSynchronize
  integer function cudastreamsynchronize(stream)
    import cuda_stream_kind
    !DIR$ IGNORE_TKR (K) stream
    integer(kind=cuda_stream_kind), value :: stream
  end function
  integer function cudastreamsynchronizenull()
  end function
end interface

end module cuda_runtime_api
