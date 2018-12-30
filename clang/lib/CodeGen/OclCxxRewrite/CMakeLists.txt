set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

# In a standard Clang+LLVM build, we need to generate intrinsics before
# building codegen. In a standalone build, LLVM is already built and we don't
# need this dependency. Furthermore, LLVM doesn't export it so we can't have
# this dependency.
set(codegen_deps intrinsics_gen)
if (CLANG_BUILT_STANDALONE)
  set(codegen_deps)
endif()

add_clang_library(clangOclCxxRewrite
  BifNameReflower.cpp
  OclCxxPrinter.cpp
  OclCxxDemangler.cpp
  OclCxxDemanglerResult.cpp

  DEPENDS
  ${codegen_deps}

  LINK_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangLex
  )
