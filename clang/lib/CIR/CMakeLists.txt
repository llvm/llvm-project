include_directories(${CLANG_CIR_MLIR_INCLUDE_DIRS})

add_subdirectory(Dialect)
add_subdirectory(CodeGen)
add_subdirectory(FrontendAction)
add_subdirectory(Interfaces)
add_subdirectory(Lowering)

add_clang_library(CIRRegisterAllDialects
  RegisterAllDialects.cpp

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRCIR
  CIROpenACCSupport
  CIROpenMPSupport
  MLIRArithDialect
  MLIRDLTIDialect
  MLIRLLVMDialect
)
