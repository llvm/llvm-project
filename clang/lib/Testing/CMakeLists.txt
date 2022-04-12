set(LLVM_LINK_COMPONENTS
  Support
  )

# Not add_clang_library: this is not part of clang's public library interface.
# Unit tests should depend on this with target_link_libraries(), rather
# than with clang_target_link_libraries().
add_llvm_library(clangTesting
  CommandLineArgs.cpp
  BUILDTREE_ONLY

  LINK_COMPONENTS
  Support
  )
