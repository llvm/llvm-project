#===============================================================================
# Lit Test Infrastructure for LLVM libc
#
# This module provides functions to set up lit-based testing for libc.
#
# Lit discovers and runs the test executables created by the
# add_libc_test() infrastructure. No separate build rules are needed.
#===============================================================================

# Guard against double inclusion
if(LIBC_LIT_TEST_RULES_INCLUDED)
  return()
endif()
set(LIBC_LIT_TEST_RULES_INCLUDED TRUE)

# Include LLVM's lit infrastructure
include(AddLLVM)

#-------------------------------------------------------------------------------
# configure_libc_lit_site_cfg()
#
# Configures the lit.site.cfg.py file from its template.
# This should be called once from libc/test/CMakeLists.txt.
#-------------------------------------------------------------------------------
function(configure_libc_lit_site_cfg)
  # Set variables for the template
  set(LIBC_SOURCE_DIR "${LIBC_SOURCE_DIR}")
  set(LIBC_BINARY_DIR "${LIBC_BUILD_DIR}")
  
  # Configure the site config file
  configure_lit_site_cfg(
    ${LIBC_SOURCE_DIR}/test/lit.site.cfg.py.in
    ${LIBC_BUILD_DIR}/test/lit.site.cfg.py
    MAIN_CONFIG
    ${LIBC_SOURCE_DIR}/test/lit.cfg.py
    PATHS
    "LLVM_SOURCE_DIR"
    "LLVM_BINARY_DIR"
    "LLVM_TOOLS_DIR"
    "LLVM_LIBS_DIR"
    "LIBC_SOURCE_DIR"
    "LIBC_BINARY_DIR"
  )
endfunction()

#-------------------------------------------------------------------------------
# add_libc_lit_testsuite()
#
# Creates a lit test suite target for a specific test directory.
#
# Usage:
#   add_libc_lit_testsuite(check-libc-ctype
#     SUITE_NAME ctype
#     TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/ctype
#     DEPENDS libc-ctype-tests
#   )
#
# Note: TEST_DIR should be the build directory where test executables are
# located, not the source directory.
#-------------------------------------------------------------------------------
function(add_libc_lit_testsuite target_name)
  cmake_parse_arguments(
    "LIT_SUITE"
    ""                    # No optional arguments
    "SUITE_NAME;TEST_DIR" # Single value arguments
    "DEPENDS"             # Multi value arguments
    ${ARGN}
  )
  
  if(NOT LIT_SUITE_TEST_DIR)
    message(FATAL_ERROR "add_libc_lit_testsuite requires TEST_DIR")
  endif()
  
  # Create the lit test target using LLVM's infrastructure
  add_lit_testsuite(${target_name}
    "Running ${LIT_SUITE_SUITE_NAME} libc tests"
    ${LIT_SUITE_TEST_DIR}
    DEPENDS ${LIT_SUITE_DEPENDS}
  )
  
  # Add to the umbrella check-libc-lit target if it exists
  if(TARGET check-libc-lit)
    add_dependencies(check-libc-lit ${target_name})
  endif()
endfunction()
