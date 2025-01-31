set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_clang_library(clangCIRLoweringDirectToLLVM
  LowerToLLVM.cpp
  )
