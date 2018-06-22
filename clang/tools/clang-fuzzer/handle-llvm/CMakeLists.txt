set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  MC
  Support
  Analysis
)

add_clang_library(clangHandleLLVM
  handle_llvm.cpp
  )
