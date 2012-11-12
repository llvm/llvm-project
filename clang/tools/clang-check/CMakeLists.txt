set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  support
  mc
  )

add_clang_executable(clang-check
  ClangCheck.cpp
  )

target_link_libraries(clang-check
  clangTooling
  clangBasic
  clangRewriteFrontend
  )

install(TARGETS clang-check
  RUNTIME DESTINATION bin)
