add_distinct_clang_unittest(IPC2978Tests
  IPC2978Test.cpp
  CLANG_LIBS
  clangIPC2978
  LLVM_COMPONENTS
  Support
  )

target_compile_definitions(IPC2978Tests PRIVATE
  LLVM_TOOLS_BINARY_DIR="${LLVM_TOOLS_BINARY_DIR}"
  )
add_dependencies(IPC2978Tests clang)
