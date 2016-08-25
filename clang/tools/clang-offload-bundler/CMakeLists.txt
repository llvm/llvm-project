set(LLVM_LINK_COMPONENTS BitWriter Core Object Support)

add_clang_executable(clang-offload-bundler
  ClangOffloadBundler.cpp
  )

set(CLANG_OFFLOAD_BUNDLER_LIB_DEPS
  clangBasic
  )
  
add_dependencies(clang clang-offload-bundler)

target_link_libraries(clang-offload-bundler
  ${CLANG_OFFLOAD_BUNDLER_LIB_DEPS}
  )

install(TARGETS clang-offload-bundler RUNTIME DESTINATION bin)
