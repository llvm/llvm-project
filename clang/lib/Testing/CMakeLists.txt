# Don't include this target in LLVMExports. FIXME: is there a cleaner way?
set(EXCLUDE_FROM_ALL ON)

# Not add_clang_library: this is not part of clang's public library interface.
# Unit tests should depend on this with target_link_libraries(), rather
# than with clang_target_link_libraries().
#
# DISABLE_PCH_REUSE: clang_target_link_libraries below propagates
# CLANG_BUILD_STATIC from Clang libraries, which conflicts with the PCH.
add_llvm_library(clangTesting
  CommandLineArgs.cpp
  TestAST.cpp

  BUILDTREE_ONLY
  DISABLE_PCH_REUSE

  LINK_COMPONENTS
  MC
  Support
  TargetParser
  )

clang_target_link_libraries(clangTesting
  PRIVATE
  clangAST
  clangBasic
  clangFrontend
  clangLex
  clangSerialization
  )

target_link_libraries(clangTesting
  PRIVATE
  llvm_gtest
  )
