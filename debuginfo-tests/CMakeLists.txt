# Debug Info tests.  These tests invoke clang to generate programs with
# various types of debug info, and then run those programs under a debugger
# such as GDB or LLDB to verify the results.

find_package(Python3 COMPONENTS Interpreter)

add_llvm_executable(check-gdb-llvm-support
        llvm-prettyprinters/gdb/llvm-support.cpp
)
target_link_libraries(check-gdb-llvm-support PRIVATE LLVMSupport)

set(DEBUGINFO_TESTS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DEBUGINFO_TESTS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(DEBUGINFO_TEST_DEPS
  clang
  FileCheck
  count
  llvm-config
  llvm-objdump
  check-gdb-llvm-support
  not
  )

if ("mlir" IN_LIST LLVM_ENABLE_PROJECTS)
  add_llvm_executable(check-gdb-mlir-support
        llvm-prettyprinters/gdb/mlir-support.cpp
  )
  target_include_directories(check-gdb-mlir-support PRIVATE
                               ${LLVM_EXTERNAL_MLIR_SOURCE_DIR}/include
                               ${LLVM_BINARY_DIR}/tools/mlir/include)
  target_link_libraries(check-gdb-mlir-support PRIVATE MLIRIR)
  list(APPEND DEBUGINFO_TEST_DEPS check-gdb-mlir-support)
  set(MLIR_SOURCE_DIR  ${LLVM_EXTERNAL_MLIR_SOURCE_DIR})
endif()

if("compiler-rt" IN_LIST LLVM_ENABLE_PROJECTS)
  # llgdb-tests/asan.c and other asan* files.
  if(TARGET asan)
    list(APPEND DEBUGINFO_TEST_DEPS asan)
  endif()
  # llgdb-tests/safestack.c
  if(TARGET safestack)
    list(APPEND DEBUGINFO_TEST_DEPS safestack)
  endif()
endif()
# Many dexter tests depend on lldb.
if("lldb" IN_LIST LLVM_ENABLE_PROJECTS)
  list(APPEND DEBUGINFO_TEST_DEPS lldb lldb-server)
endif()

# The Windows builder scripts pass -fuse-ld=lld.
if (WIN32)
  set(DEBUGINFO_TEST_DEPS ${DEBUGINFO_TEST_DEPS} lld)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

add_lit_testsuite(check-debuginfo "Running debug info integration tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${DEBUGINFO_TEST_DEPS}
  )

# Add check-debuginfo-* targets.
add_lit_testsuites(DEBUGINFO ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${DEBUGINFO_TEST_DEPS}
  )

set_target_properties(check-debuginfo PROPERTIES FOLDER "Debug info tests")
