set(CLANGD_TEST_DEPS
  clangd
  ClangdTests
  clangd-indexer
  clangd-remap
  split-file
  )

if (LLVM_INCLUDE_BENCHMARKS)
  list(APPEND CLANGD_TEST_DEPS IndexBenchmark)
endif()

if(CLANGD_BUILD_XPC)
  list(APPEND CLANGD_TEST_DEPS clangd-xpc-test-client)
  list(APPEND CLANGD_TEST_DEPS ClangdXpcUnitTests)
endif()

if(CLANGD_BUILD_DEXP)
  # No tests for it, but we should still make sure they build.
  list(APPEND CLANGD_TEST_DEPS dexp)
endif()

if(CLANGD_ENABLE_REMOTE)
  list(APPEND CLANGD_TEST_DEPS clangd-index-server clangd-index-server-monitor)
endif()

foreach(dep FileCheck count not llvm-config)
  if(TARGET ${dep})
    list(APPEND CLANGD_TEST_DEPS ${dep})
  endif()
endforeach()

if (NOT WIN32 OR NOT LLVM_LINK_LLVM_DYLIB)
  llvm_add_library(
      ClangdFeatureModuleExample
      MODULE FeatureModuleExample.cpp
      PLUGIN_TOOL clangd
      PARTIAL_SOURCES_INTENDED)
  llvm_add_library(
      ClangdStandaloneTweakExample
      MODULE StandaloneTweakExample.cpp
      PLUGIN_TOOL clangd
      PARTIAL_SOURCES_INTENDED)
endif()

foreach(target ClangdFeatureModuleExample ClangdStandaloneTweakExample)
  if(TARGET ${target})
    list(APPEND CLANGD_TEST_DEPS ${target})
    target_include_directories(${target} PUBLIC BEFORE
      "${CLANG_TOOLS_SOURCE_DIR}/clangd"
      "${CMAKE_CURRENT_BINARY_DIR}/.."
    )
    if(CLANG_PLUGIN_SUPPORT AND (WIN32 OR CYGWIN))
      set(LLVM_LINK_COMPONENTS
        Support
      )
    endif()
  endif()
endforeach()

llvm_canonicalize_cmake_booleans(CLANG_PLUGIN_SUPPORT)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

# Copy an empty compile_flags.txt to make sure tests don't pick up arbitrary
# commands from parents.
file(
  TOUCH
  ${CMAKE_CURRENT_BINARY_DIR}/compile_flags.txt
)

add_lit_testsuite(check-clangd "Running the Clangd regression tests"
  # clangd doesn't put unittest configs in test/unit like every other project.
  # Because of that, this needs to pass two folders here, while every other
  # project only needs to pass CMAKE_CURRENT_BINARY_DIR.
  ${CMAKE_CURRENT_BINARY_DIR}/../unittests;${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CLANGD_TEST_DEPS})
