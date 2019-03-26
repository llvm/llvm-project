if (NOT DEFINED CLANGD_BUILD_XPC)
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CLANGD_BUILD_XPC_DEFAULT ON)
  else ()
    set(CLANGD_BUILD_XPC_DEFAULT OFF)
  endif ()

  set(CLANGD_BUILD_XPC ${CLANGD_BUILD_XPC_DEFAULT} CACHE BOOL "Build XPC Support For Clangd." FORCE)

  unset(CLANGD_BUILD_XPC_DEFAULT)
endif ()

add_subdirectory(clang-apply-replacements)
add_subdirectory(clang-reorder-fields)
add_subdirectory(modularize)
add_subdirectory(clang-tidy)
add_subdirectory(clang-tidy-vs)

add_subdirectory(clang-change-namespace)
add_subdirectory(clang-doc)
add_subdirectory(clang-include-fixer)
add_subdirectory(clang-move)
add_subdirectory(clang-query)
add_subdirectory(clangd)
add_subdirectory(pp-trace)
add_subdirectory(tool-template)

# Add the common testsuite after all the tools.
if(CLANG_INCLUDE_TESTS)
add_subdirectory(test)
add_subdirectory(unittests)
endif()

option(CLANG_TOOLS_EXTRA_INCLUDE_DOCS "Generate build targets for the Clang Extra Tools docs."
  ${LLVM_INCLUDE_DOCS})
if( CLANG_TOOLS_EXTRA_INCLUDE_DOCS )
  add_subdirectory(docs)
endif()

