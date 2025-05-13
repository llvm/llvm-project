add_clang_unittest(CIRUnitTests
  PointerLikeTest.cpp
  LLVM_COMPONENTS
  Core

  LINK_LIBS
  MLIRCIR
  CIROpenACCSupport
  MLIRIR
  MLIROpenACCDialect
  )
