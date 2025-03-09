set(
  LLVM_LINK_COMPONENTS
  Core
  Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangCIR
  CIRGenerator.cpp
  CIRGenDecl.cpp
  CIRGenExpr.cpp
  CIRGenExprConstant.cpp
  CIRGenExprScalar.cpp
  CIRGenFunction.cpp
  CIRGenModule.cpp
  CIRGenStmt.cpp
  CIRGenTypes.cpp

  DEPENDS
  MLIRCIR
  MLIRCIROpInterfacesIncGen
  ${dialect_libs}

  LINK_LIBS
  clangAST
  clangBasic
  clangLex
  ${dialect_libs}
  MLIRCIR
  MLIRCIRInterfaces
)
