set(LLVM_LINK_COMPONENTS
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangCIRFrontendAction
  CIRGenAction.cpp

  LINK_LIBS
  MLIRCIR
  MLIRIR
  )
