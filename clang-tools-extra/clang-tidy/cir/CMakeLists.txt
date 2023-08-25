set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/.. )
include_directories( ${LLVM_MAIN_SRC_DIR}/../mlir/include )
include_directories( ${CMAKE_BINARY_DIR}/tools/mlir/include )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_clang_library(clangTidyCIRModule
  Lifetime.cpp
  CIRTidyModule.cpp

  LINK_LIBS
  clangASTMatchers
  clangCIR
  clangFrontend
  clangSerialization
  clangTidy
  clangTidyUtils
  ${dialect_libs}
  MLIRCIR
  MLIRCIRTransforms
  MLIRAffineToStandard
  MLIRAnalysis
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRSCFToControlFlow
  MLIRFuncToLLVM
  MLIRSupport
  MLIRMemRefDialect
  MLIRTargetLLVMIRExport
  MLIRTransforms

  DEPENDS
  omp_gen
  )

clang_target_link_libraries(clangTidyCIRModule
  PRIVATE
  clangAnalysis
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangTooling
  clangToolingCore
  )
