set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  support
  )

add_clang_executable(clang-func-mapping
  ClangFnMapGen.cpp
  )

target_link_libraries(clang-func-mapping
  PRIVATE
  clangAST
  clangBasic
  clangCrossTU
  clangFrontend
  clangSerialization
  clangTooling
  )

install(TARGETS clang-func-mapping
  RUNTIME DESTINATION bin)
