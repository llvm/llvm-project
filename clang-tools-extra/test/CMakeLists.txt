# Test runner infrastructure for Clang-based tools. This configures the Clang
# test trees for use by Lit, and delegates to LLVM's lit test handlers.
#
# Note that currently we don't support stand-alone builds of Clang, you must
# be building Clang from within a combined LLVM+Clang checkout..

set(CLANG_TOOLS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(CLANG_TOOLS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/..")

llvm_canonicalize_cmake_booleans(
  CLANG_TIDY_ENABLE_STATIC_ANALYZER
  CLANG_PLUGIN_SUPPORT
  LLVM_INSTALL_TOOLCHAIN_ONLY
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
  )

set(CLANG_TOOLS_TEST_DEPS
  # For the clang-apply-replacements test that uses clang-rename.
  clang-rename

  # For the clang-doc tests that emit bitcode files.
  llvm-bcanalyzer

  # Individual tools we test.
  clang-apply-replacements
  clang-change-namespace
  clang-doc
  clang-include-fixer
  clang-move
  clang-query
  clang-reorder-fields
  find-all-symbols
  modularize
  pp-trace

  # Unit tests
  ExtraToolsUnitTests

  # clang-tidy tests require it.
  clang-resource-headers

  clang-tidy
  # Clang-tidy tests need clang for building modules.
  clang
)

# Add lit test dependencies.
set(LLVM_UTILS_DEPS
  FileCheck count not
)
foreach(dep ${LLVM_UTILS_DEPS})
  if(TARGET ${dep})
    list(APPEND CLANG_TOOLS_TEST_DEPS ${dep})
  endif()
endforeach()

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  if (NOT WIN32 OR NOT LLVM_LINK_LLVM_DYLIB)
    llvm_add_library(
        CTTestTidyModule
        MODULE clang-tidy/CTTestTidyModule.cpp
        PLUGIN_TOOL clang-tidy
        DEPENDS clang-tidy-headers)
  endif()

  if(CLANG_BUILT_STANDALONE)
    # LLVMHello library is needed below
    if (EXISTS ${LLVM_MAIN_SRC_DIR}/lib/Transforms/Hello
       AND NOT TARGET LLVMHello)
      add_subdirectory(${LLVM_MAIN_SRC_DIR}/lib/Transforms/Hello
        lib/Transforms/Hello)
    endif()
  endif()

  if(TARGET CTTestTidyModule)
      list(APPEND CLANG_TOOLS_TEST_DEPS CTTestTidyModule LLVMHello)
      target_include_directories(CTTestTidyModule PUBLIC BEFORE "${CLANG_TOOLS_SOURCE_DIR}")
      if(CLANG_PLUGIN_SUPPORT AND (WIN32 OR CYGWIN))
        set(LLVM_LINK_COMPONENTS
          Support
        )
      endif()
  endif()
endif()

add_lit_testsuites(CLANG-EXTRA ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${CLANG_TOOLS_TEST_DEPS}
  )
