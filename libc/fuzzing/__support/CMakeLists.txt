add_libc_fuzzer(
  uint_fuzz
  SRCS
    uint_fuzz.cpp
  DEPENDS
    libc.src.__support.big_int
)

add_libc_fuzzer(
  hashtable_fuzz
  SRCS
    hashtable_fuzz.cpp
  DEPENDS
    libc.src.__support.HashTable.table
)

add_libc_fuzzer(
  hashtable_opt_fuzz
  SRCS
    hashtable_fuzz.cpp
  DEPENDS
    libc.src.__support.HashTable.table
  COMPILE_OPTIONS
    -D__LIBC_EXPLICIT_SIMD_OPT
) 

# TODO: FreeListHeap uses the _end symbol which conflicts with the _end symbol
# defined by GPU start.cpp files so for now we exclude this fuzzer on GPU.
if(LLVM_LIBC_FULL_BUILD AND NOT LIBC_TARGET_OS_IS_GPU)
  add_libc_fuzzer(
    freelist_heap_fuzz
    SRCS
      fake_heap.s
      freelist_heap_fuzz.cpp
    DEPENDS
      libc.src.__support.freelist_heap
  )
  # TODO(#119995): Remove this once sccache on Windows no longer requires
  # the use of -DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded.
  get_fq_target_name(freelist_heap_fuzz freelist_heap_fuzz_target_name)
  set_target_properties(
    ${freelist_heap_fuzz_target_name}
    PROPERTIES
      MSVC_DEBUG_INFORMATION_FORMAT ""
  )
endif()
