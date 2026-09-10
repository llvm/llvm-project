add_executable(inter-ged-dump ged-dump.cpp)
target_link_libraries(inter-ged-dump PRIVATE InterGED)

add_executable(inter-alias-dump alias-dump.cpp)
target_link_libraries(inter-alias-dump PRIVATE
  InterXeMachine
  MLIRFuncDialect
  MLIRParser)

add_executable(inter-timing-dump timing-dump.cpp)
target_link_libraries(inter-timing-dump PRIVATE
  InterXeMachine
  MLIRFuncDialect
  MLIRParser)

add_executable(inter-target-info target-info.cpp)
target_link_libraries(inter-target-info PRIVATE InterXeMachine)

add_executable(inter-compile-api-test compile-api.cpp)
target_link_libraries(inter-compile-api-test PRIVATE
  InterCompiler
  LLVMAsmParser)

set(INTER_TEST_SOURCE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(INTER_TEST_EXEC_ROOT ${CMAKE_CURRENT_BINARY_DIR})
set(INTER_TEST_IS_INTEGRATION 0)
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(INTER_TEST_DEPENDS
  FileCheck
  inter-compile-api-test
  inter-alias-dump
  inter-ged-dump
  inter-opt
  inter-target-info
  inter-timing-dump
  inter-translate
  llvm-objcopy
  llvm-readobj)

add_lit_testsuite(check-inter "Running the inter regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${INTER_TEST_DEPENDS})
set_target_properties(check-inter PROPERTIES FOLDER "Tests")

if(INTER_INCLUDE_INTEGRATION_TESTS)
  find_program(INTER_OCLOC_EXECUTABLE NAMES ocloc REQUIRED)
  get_filename_component(INTER_LLVM_BUILD_ROOT
    "${LLVM_TOOLS_BINARY_DIR}/.." ABSOLUTE)
  find_path(INTER_OFFLOAD_INCLUDE_DIR OffloadAPI.h
    HINTS
      "${INTER_LLVM_BUILD_ROOT}/runtimes/runtimes-bins/offload/liboffload/API"
    REQUIRED)
  find_library(INTER_OFFLOAD_LIBRARY LLVMOffload
    HINTS
      "${INTER_LLVM_BUILD_ROOT}/lib/${LLVM_HOST_TRIPLE}"
      "${INTER_LLVM_BUILD_ROOT}/lib"
    REQUIRED)

  add_library(InterLLVMOffload SHARED IMPORTED)
  set_target_properties(InterLLVMOffload PROPERTIES
    IMPORTED_LOCATION "${INTER_OFFLOAD_LIBRARY}")

  add_executable(inter-runner ${PROJECT_SOURCE_DIR}/launcher.cpp)
  target_include_directories(inter-runner PRIVATE ${INTER_OFFLOAD_INCLUDE_DIR})
  target_link_libraries(inter-runner PRIVATE InterLLVMOffload)
  add_executable(inter-matmul-runner matmul-runner.cpp)
  target_include_directories(inter-matmul-runner PRIVATE
    ${INTER_OFFLOAD_INCLUDE_DIR})
  target_link_libraries(inter-matmul-runner PRIVATE InterLLVMOffload)
  get_filename_component(INTER_OFFLOAD_LIBRARY_DIR
    "${INTER_OFFLOAD_LIBRARY}" DIRECTORY)
  set_target_properties(inter-runner PROPERTIES
    BUILD_RPATH "${INTER_OFFLOAD_LIBRARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Integration")
  set_target_properties(inter-matmul-runner PROPERTIES
    BUILD_RPATH "${INTER_OFFLOAD_LIBRARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Integration")

  set(INTER_TEST_SOURCE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/Integration)
  set(INTER_TEST_EXEC_ROOT ${CMAKE_CURRENT_BINARY_DIR}/Integration)
  set(INTER_TEST_IS_INTEGRATION 1)
  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/Integration/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

  add_lit_testsuite(check-inter-integration
    "Running the inter Intel Xe integration tests"
    ${CMAKE_CURRENT_BINARY_DIR}/Integration
    DEPENDS ${INTER_TEST_DEPENDS} inter-runner inter-matmul-runner)
  set_target_properties(check-inter-integration PROPERTIES FOLDER "Tests")
endif()
