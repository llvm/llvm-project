if(CMAKE_COMPILER_IS_GNUCXX OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions")
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_library(RuntimeTesting
  testing.cpp
  )

add_executable(format-test
  format.cpp
)

target_link_libraries(format-test
  FortranRuntime
  RuntimeTesting
  LLVMSupport
)

add_test(NAME Format COMMAND format-test)

add_executable(hello-world
  hello.cpp
)

target_link_libraries(hello-world
  FortranRuntime
  RuntimeTesting
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
  FortranRuntime
  RuntimeTesting
  LLVMSupport
)

add_test(NAME ListInput COMMAND list-input-test)
