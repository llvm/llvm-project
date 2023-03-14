if(NOT FLANG_BUILD_EXAMPLES)
  set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL ON)
endif()

add_subdirectory(ExternalHelloWorld)
add_subdirectory(PrintFlangFunctionNames)
add_subdirectory(FlangOmpReport)
