# Cross project tests, for tests that require access to multiple projects across
# LLVM (e.g. clang, lld and lldb).
# The subset inside debuginfo-tests invoke clang to generate programs with
# various types of debug info, and then run those programs under a debugger
# such as GDB or LLDB to verify the results.
set(LLVM_SUBPROJECT_TITLE "Cross-Project")

find_package(Python3 COMPONENTS Interpreter)

add_llvm_executable(check-gdb-llvm-support
        debuginfo-tests/llvm-prettyprinters/gdb/llvm-support.cpp
)
target_link_libraries(check-gdb-llvm-support PRIVATE LLVMSupport)

set(CROSS_PROJECT_TESTS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(CROSS_PROJECT_TESTS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(CROSS_PROJECT_TEST_DEPS
  FileCheck
  check-gdb-llvm-support
  count
  llvm-ar
  llvm-config
  llvm-dwarfdump
  llvm-lto2
  llvm-objdump
  llvm-profdata
  not
  opt
  split-file
  )

if ("clang" IN_LIST LLVM_ENABLE_PROJECTS)
  list(APPEND CROSS_PROJECT_TEST_DEPS clang)
endif()

if ("mlir" IN_LIST LLVM_ENABLE_PROJECTS)
  add_llvm_executable(check-gdb-mlir-support
        debuginfo-tests/llvm-prettyprinters/gdb/mlir-support.cpp
  )
  target_include_directories(check-gdb-mlir-support PRIVATE
                               ${LLVM_EXTERNAL_MLIR_SOURCE_DIR}/include
                               ${LLVM_BINARY_DIR}/tools/mlir/include)
  target_link_libraries(check-gdb-mlir-support PRIVATE MLIRIR)
  list(APPEND CROSS_PROJECT_TEST_DEPS check-gdb-mlir-support)
  set(MLIR_SOURCE_DIR  ${LLVM_EXTERNAL_MLIR_SOURCE_DIR})
endif()

if("compiler-rt" IN_LIST LLVM_ENABLE_PROJECTS)
  # llgdb-tests/asan.c and other asan* files.
  if(TARGET asan)
    list(APPEND CROSS_PROJECT_TEST_DEPS asan)
  endif()
  # llgdb-tests/safestack.c
  if(TARGET safestack)
    list(APPEND CROSS_PROJECT_TEST_DEPS safestack)
  endif()
endif()
# Many dexter tests depend on lldb.
if("lldb" IN_LIST LLVM_ENABLE_PROJECTS)
  list(APPEND CROSS_PROJECT_TEST_DEPS lldb lldb-server)
endif()

if ("lld" IN_LIST LLVM_ENABLE_PROJECTS)
  list(APPEND CROSS_PROJECT_TEST_DEPS lld)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

add_lit_testsuite(check-cross-project "Running cross-project tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CROSS_PROJECT_TEST_DEPS}
  )

# Add alias for debuginfo test subset.
add_lit_testsuite(check-debuginfo "Running debuginfo tests"
  ${CMAKE_CURRENT_BINARY_DIR}/debuginfo-tests
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS ${CROSS_PROJECT_TEST_DEPS}
  )

# Add alias for intrinsic header test subset.
add_lit_testsuite(check-intrinsic-headers "Running intrinsic header tests"
  ${CMAKE_CURRENT_BINARY_DIR}/intrinsic-header-tests
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS ${CROSS_PROJECT_TEST_DEPS}
  )

# AMDGPU tests.
add_lit_testsuite(check-cross-amdgpu "Running AMDGPU cross-project tests"
  ${CMAKE_CURRENT_BINARY_DIR}/amdgpu
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS clang
  )

# DTLTO tests.
add_lit_testsuite(check-cross-dtlto "Running DTLTO cross-project tests"
  ${CMAKE_CURRENT_BINARY_DIR}/dtlto
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS ${CROSS_PROJECT_TEST_DEPS}
  )

# Add check-cross-project-* targets.
add_lit_testsuites(CROSS_PROJECT ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CROSS_PROJECT_TEST_DEPS}
  )
