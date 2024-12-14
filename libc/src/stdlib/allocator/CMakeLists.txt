if(LIBC_TARGET_OS_IS_GPU)
  message(STATUS "Skipping allocators for GPU targets.")
  return()
endif()

# TODO: we should really use option to dispatch the allocator. Currently, there are
# other behaviors control the allocator selection. We simulate the existing behavior
# prior to the modularization of allocator but this should be cleaned up.

if(LLVM_LIBC_INCLUDE_SCUDO)
  set(DEFAULT_ALLOCATOR_VARIANT "scudo")
elseif(LIBC_TARGET_OS_IS_BAREMETAL)
  set(DEFAULT_ALLOCATOR_VARIANT "freelist")
else()
  set(DEFAULT_ALLOCATOR_VARIANT "external")
endif()

set(LIBC_ALLOCATOR_VARIANT ${DEFAULT_ALLOCATOR_VARIANT} CACHE STRING
  "The allocator variant to use. Options are: scudo, freelist, external.")

message(STATUS "Using ${LIBC_ALLOCATOR_VARIANT} allocator.")

if (LIBC_ALLOCATOR_VARIANT STREQUAL "scudo")
  if (NOT LLVM_LIBC_INCLUDE_SCUDO)
    message(FATAL_ERROR "SCUDO allocator is not enabled.")
  endif()
  add_subdirectory(scudo)
elseif (LIBC_ALLOCATOR_VARIANT STREQUAL "freelist")
  add_subdirectory(freelist)
elseif (LIBC_ALLOCATOR_VARIANT STREQUAL "external")
  add_subdirectory(external)
else()
  message(FATAL_ERROR "Unknown allocator variant: ${LIBC_ALLOCATOR_VARIANT}")
endif()
