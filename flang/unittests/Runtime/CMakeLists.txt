set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# RuntimeTesting needs exceptions enabled
set(LLVM_REQUIRES_EH ON)
set(LLVM_REQUIRES_RTTI ON)
add_library(RuntimeTesting
  testing.cpp
)
llvm_update_compile_flags(RuntimeTesting)

add_executable(format-test
  format.cpp
)

target_link_libraries(format-test
  RuntimeTesting
  FortranRuntime
  LLVMSupport
)

add_test(NAME Format COMMAND format-test)

add_executable(hello-world
  hello.cpp
)

target_link_libraries(hello-world
  RuntimeTesting
  FortranRuntime
  LLVMSupport
)

add_test(NAME HelloWorld COMMAND hello-world)

add_executable(external-hello-world
  external-hello.cpp
)

target_link_libraries(external-hello-world
  FortranRuntime
  LLVMSupport
)

add_executable(list-input-test
  list-input.cpp
)

target_link_libraries(list-input-test
  RuntimeTesting
  FortranRuntime
  LLVMSupport
)

add_test(NAME ListInput COMMAND list-input-test)

add_executable(character-test
  character.cpp
)

target_link_libraries(character-test
  RuntimeTesting
  FortranRuntime
  LLVMSupport
)

add_test(NAME CharacterTest COMMAND character-test)
