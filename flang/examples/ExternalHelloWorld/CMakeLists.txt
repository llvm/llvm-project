set(LLVM_LINK_COMPONENTS
  FortranRuntime
  )

# This test is not run by default as it requires input.
add_llvm_example(external-hello-world
  external-hello.cpp
)
