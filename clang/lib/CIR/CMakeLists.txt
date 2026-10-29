include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${LLVM_BINARY_DIR}/tools/mlir/include)

add_subdirectory(Dialect)
add_subdirectory(CodeGen)
add_subdirectory(FrontendAction)
add_subdirectory(Interfaces)
add_subdirectory(Lowering)

add_clang_library(CIRRegisterAllDialects
  RegisterAllDialects.cpp

  DEPENDS
  MLIRArithOpsIncGen

  LINK_LIBS PUBLIC
  MLIRCIR
  CIROpenACCSupport
  CIROpenMPSupport
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRLLVMDialect
)
