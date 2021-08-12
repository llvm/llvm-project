if(NOT FLANG_BUILD_EXAMPLES)
  set(EXCLUDE_FROM_ALL ON)
endif()

# This test is not run by default as it requires input.
add_executable(external-hello-world
  external-hello.cpp
)

target_link_libraries(external-hello-world
  FortranRuntime
)

add_subdirectory(HelloWorld)
