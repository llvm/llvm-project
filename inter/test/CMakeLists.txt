add_executable(inter-ged-dump ged-dump.cpp)
target_link_libraries(inter-ged-dump PRIVATE InterGED)

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
  inter-ged-dump
  inter-opt
  inter-translate)

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
  get_filename_component(INTER_OFFLOAD_LIBRARY_DIR
    "${INTER_OFFLOAD_LIBRARY}" DIRECTORY)
  set_target_properties(inter-runner PROPERTIES
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
    DEPENDS ${INTER_TEST_DEPENDS} inter-runner)
  set_target_properties(check-inter-integration PROPERTIES FOLDER "Tests")
endif()
