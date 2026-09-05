add_libc_testsuite(libc-elf-tests)

# map_segments_test writes a backing file to map.
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testdata)

add_libc_test(
  dynamic_test
  SUITE
    libc-support-tests
  SRCS
    dynamic_test.cpp
  DEPENDS
    libc.src.__support.elf.dynamic
)

add_libc_test(
  hash_test
  SUITE
    libc-support-tests
  SRCS
    hash_test.cpp
  DEPENDS
    libc.src.__support.elf.hash
)

add_libc_test(
  symbol_test
  SUITE
    libc-support-tests
  SRCS
    symbol_test.cpp
  DEPENDS
    libc.src.__support.elf.symbol
)

add_libc_test(
  relocation_test
  SUITE
    libc-support-tests
  SRCS
    relocation_test.cpp
  DEPENDS
    libc.src.__support.elf.relocation
)

add_libc_test(
  module_test
  SUITE
    libc-support-tests
  SRCS
    module_test.cpp
  DEPENDS
    libc.src.__support.elf.module
)

add_libc_test(
  tls_layout_test
  SUITE
    libc-support-tests
  SRCS
    tls_layout_test.cpp
  DEPENDS
    libc.src.__support.elf.tls_layout
)


add_libc_test(
  load_layout_test
  SUITE
    libc-support-tests
  SRCS
    load_layout_test.cpp
  DEPENDS
    libc.src.__support.elf.load_layout
)

add_libc_test(
  map_segments_test
  SUITE
    libc-support-tests
  SRCS
    map_segments_test.cpp
  DEPENDS
    libc.hdr.fcntl_macros
    libc.src.__support.OSUtil.linux.syscall_wrappers.close
    libc.src.__support.OSUtil.linux.syscall_wrappers.open
    libc.src.__support.OSUtil.linux.syscall_wrappers.write
    libc.src.__support.elf.map_segments
)

if(LIBC_ENABLE_SHARED)
  add_libc_test(
    load_module_test
    SUITE
      libc-support-tests
    SRCS
      load_module_test.cpp
    COMPILE_OPTIONS
      -DLIBC_TEST_SHARED_OBJECT="$<TARGET_FILE:libc-shared>"
    DEPENDS
      libc.src.__support.elf.load_module
  )
endif()

if(LIBC_ENABLE_SHARED)
  add_libc_test(
    bind_test
    SUITE
      libc-support-tests
    SRCS
      bind_test.cpp
    COMPILE_OPTIONS
      -DLIBC_TEST_SHARED_OBJECT="$<TARGET_FILE:libc-shared>"
      -DLIBM_TEST_SHARED_OBJECT="$<TARGET_FILE:libm-shared>"
    DEPENDS
      libc.src.__support.elf.bind
      libc.src.__support.elf.load_module
  )
endif()

if(LIBC_ENABLE_SHARED)
  add_libc_test(
    tls_block_test
    SUITE
      libc-support-tests
    SRCS
      tls_block_test.cpp
    COMPILE_OPTIONS
      -DLIBC_TEST_SHARED_OBJECT="$<TARGET_FILE:libc-shared>"
    DEPENDS
      libc.src.__support.elf.load_module
      libc.src.__support.elf.tls_block
  )
endif()

if(LIBC_ENABLE_SHARED)
  add_libc_test(
    registry_test
    SUITE
      libc-support-tests
    SRCS
      registry_test.cpp
    COMPILE_OPTIONS
      -DLIBC_TEST_SHARED_OBJECT="$<TARGET_FILE:libc-shared>"
      -DLIBM_TEST_SHARED_OBJECT="$<TARGET_FILE:libm-shared>"
    DEPENDS
      libc.src.__support.elf.registry
  )
endif()

add_libc_test(
  startup_stack_test
  SUITE
    libc-support-tests
  SRCS
    startup_stack_test.cpp
  DEPENDS
    libc.src.__support.elf.startup_stack
)

add_libc_test(
  run_path_test
  SUITE
    libc-support-tests
  SRCS
    run_path_test.cpp
  DEPENDS
    libc.src.__support.CPP.string_view
    libc.src.__support.elf.run_path
)
