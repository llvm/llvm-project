if(NOT CLANG_BUILD_EXAMPLES)
  set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL ON)
  set(EXCLUDE_FROM_ALL ON)
endif()

add_subdirectory(clang-interpreter)
add_subdirectory(PrintFunctionNames)
add_subdirectory(AnnotateFunctions)
