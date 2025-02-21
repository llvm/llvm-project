# This test is not run by default as it requires input.
add_llvm_example(external-hello-world
  ${FLANG_RT_SOURCE_DIR}/examples/ExternalHelloWorld/external-hello.cpp
)

target_link_libraries(external-hello-world
  PRIVATE
  flang_rt.runtime
  )
