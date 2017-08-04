add_subdirectory(clang-apply-replacements)
add_subdirectory(clang-reorder-fields)
add_subdirectory(modularize)
if(CLANG_ENABLE_STATIC_ANALYZER)
add_subdirectory(clang-tidy)
add_subdirectory(clang-tidy-vs)
endif()

add_subdirectory(change-namespace)
add_subdirectory(clang-query)
add_subdirectory(clang-move)
add_subdirectory(clangd)
add_subdirectory(include-fixer)
add_subdirectory(pp-trace)
add_subdirectory(tool-template)

# Add the common testsuite after all the tools.
# TODO: Support tests with more granularity when features are off?
if(CLANG_ENABLE_STATIC_ANALYZER AND CLANG_INCLUDE_TESTS)
add_subdirectory(test)
add_subdirectory(unittests)
endif()

option(CLANG_TOOLS_EXTRA_INCLUDE_DOCS "Generate build targets for the Clang Extra Tools docs."
  ${LLVM_INCLUDE_DOCS})
if( CLANG_TOOLS_EXTRA_INCLUDE_DOCS )
  add_subdirectory(docs)
endif()

