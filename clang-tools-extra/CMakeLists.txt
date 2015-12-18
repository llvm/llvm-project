add_subdirectory(clang-apply-replacements)
add_subdirectory(clang-rename)
add_subdirectory(modularize)
if(CLANG_ENABLE_STATIC_ANALYZER)
add_subdirectory(clang-tidy)
endif()

add_subdirectory(clang-query)
add_subdirectory(pp-trace)
add_subdirectory(tool-template)

# Add the common testsuite after all the tools.
# TODO: Support tests with more granularity when features are off?
if(CLANG_ENABLE_STATIC_ANALYZER AND CLANG_INCLUDE_TESTS)
add_subdirectory(test)
add_subdirectory(unittests)
endif()
