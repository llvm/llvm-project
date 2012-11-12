set( LLVM_LINK_COMPONENTS
  support
  mc
  )

add_clang_executable(c-arcmt-test
  c-arcmt-test.c
  )

target_link_libraries(c-arcmt-test
  libclang
  )

set_target_properties(c-arcmt-test
  PROPERTIES
  LINKER_LANGUAGE CXX)
