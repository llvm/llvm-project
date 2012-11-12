set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  asmparser
  support
  mc
  )

add_clang_executable(arcmt-test
  arcmt-test.cpp
  )

target_link_libraries(arcmt-test
  clangARCMigrate
  clangEdit
  clangRewriteCore
  )
