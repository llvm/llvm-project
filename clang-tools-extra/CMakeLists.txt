add_subdirectory(remove-cstr-calls)
add_subdirectory(tool-template)
add_subdirectory(cpp11-migrate)
add_subdirectory(modularize)

# Add the common testsuite after all the tools.
add_subdirectory(test)
add_subdirectory(unittests)
