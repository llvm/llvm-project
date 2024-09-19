set(LLVM_SUBPROJECT_TITLE "Clang Tools Extra")

include(CMakeDependentOption)
include(GNUInstallDirs)

option(CLANG_TIDY_ENABLE_STATIC_ANALYZER
  "Include static analyzer checks in clang-tidy" ON)

if(CLANG_INCLUDE_TESTS)
  umbrella_lit_testsuite_begin(check-clang-tools)

  option(CLANG_TOOLS_TEST_USE_VG "Run Clang tools' tests under Valgrind" OFF)
  if(CLANG_TOOLS_TEST_USE_VG)
    set_property(GLOBAL APPEND PROPERTY LLVM_CLANG_TOOLS_LIT_EXTRA_ARGS "--vg")
  endif()
endif()

add_subdirectory(clang-apply-replacements)
add_subdirectory(clang-reorder-fields)
add_subdirectory(modularize)
add_subdirectory(clang-tidy)

add_subdirectory(clang-change-namespace)
add_subdirectory(clang-doc)
add_subdirectory(clang-include-fixer)
add_subdirectory(clang-move)
add_subdirectory(clang-query)
add_subdirectory(include-cleaner)
add_subdirectory(pp-trace)
add_subdirectory(tool-template)

option(CLANG_TOOLS_EXTRA_INCLUDE_DOCS "Generate build targets for the Clang Extra Tools docs."
  ${LLVM_INCLUDE_DOCS})
if( CLANG_TOOLS_EXTRA_INCLUDE_DOCS )
  add_subdirectory(docs)
endif()

# clangd has its own CMake tree. It requires threads.
CMAKE_DEPENDENT_OPTION(CLANG_ENABLE_CLANGD "Build clangd language server" ON
                       "LLVM_ENABLE_THREADS" OFF)
if (CLANG_ENABLE_CLANGD)
  add_subdirectory(clangd)
endif()

# Add the common testsuite after all the tools.
if(CLANG_INCLUDE_TESTS)
  add_subdirectory(test)
  add_subdirectory(unittests)
  umbrella_lit_testsuite_end(check-clang-tools)
endif()
