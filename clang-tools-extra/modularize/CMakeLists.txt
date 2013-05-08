set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  support
  mc
  )

add_clang_executable(modularize
  Modularize.cpp
  )

target_link_libraries(modularize
  clangTooling
  clangBasic
  clangRewriteFrontend
  )
