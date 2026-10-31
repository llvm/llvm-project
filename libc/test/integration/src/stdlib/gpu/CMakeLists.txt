add_libc_testsuite(libc-stdlib-gpu-integration-tests PARENT libc-stdlib-integration-tests)

# TODO: Test on NVPTX, requires CUDA VMEM API.
if(NOT LIBC_TARGET_ARCHITECTURE_IS_NVPTX)
  add_integration_test(
    malloc
    SUITE
      libc-stdlib-gpu-integration-tests
    SRCS
      malloc.cpp
    DEPENDS
      libc.src.stdlib.malloc
      libc.src.stdlib.free
    LOADER_ARGS
      --threads 256
      --blocks 1024
  )

  add_integration_test(
    realloc
    SUITE
      libc-stdlib-gpu-integration-tests
    SRCS
      realloc.cpp
    DEPENDS
      libc.src.stdlib.malloc
      libc.src.stdlib.free
      libc.src.stdlib.realloc
    LOADER_ARGS
      --threads 256
      --blocks 1024
  )

  add_integration_test(
    aligned_alloc
    SUITE
      libc-stdlib-gpu-integration-tests
    SRCS
      aligned_alloc.cpp
    DEPENDS
      libc.src.stdlib.aligned_alloc
      libc.src.stdlib.malloc
      libc.src.stdlib.free
    LOADER_ARGS
      --threads 256
      --blocks 128
  )

  add_integration_test(
    malloc_stress
    SUITE
      libc-stdlib-gpu-integration-tests
    SRCS
      malloc_stress.cpp
    DEPENDS
      libc.src.stdlib.malloc
      libc.src.stdlib.free
    LOADER_ARGS
      --threads 256
      --blocks 2048
  )
endif()
