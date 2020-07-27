set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# RuntimeTesting needs exceptions enabled
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)
add_library(RuntimeTesting
  testing.cpp
)
llvm_update_compile_flags(RuntimeTesting)

target_link_libraries(RuntimeTesting
  FortranRuntime
  LLVMSupport
)

add_flang_nongtest_unittest(format
  RuntimeTesting
  FortranRuntime
  LLVMSupport
)

add_flang_nongtest_unittest(hello
  RuntimeTesting
  FortranRuntime
  LLVMSupport
)

# This test is not run by default as it requires input.
add_executable(external-hello-world
  external-hello.cpp
)

target_link_libraries(external-hello-world
  FortranRuntime
  LLVMSupport
)

add_flang_nongtest_unittest(external-io
  RuntimeTesting
  FortranRuntime
  LLVMSupport
)

add_flang_nongtest_unittest(list-input
  RuntimeTesting
  FortranRuntime
  LLVMSupport
)

add_flang_nongtest_unittest(character
  RuntimeTesting
  FortranRuntime
  LLVMSupport
)
