set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  MC
  Support
  Analysis
  )

# Depend on LLVM IR intrinsic generation.
set(handle_llvm_deps intrinsics_gen)
if (CLANG_BUILT_STANDALONE)
  set(handle_llvm_deps)
endif()

add_clang_library(clangHandleLLVM
  handle_llvm.cpp

  DEPENDS
  ${handle_llvm_deps}
  )
