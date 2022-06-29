set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/.. )
include_directories( ${LLVM_MAIN_SRC_DIR}/../mlir/include )
include_directories( ${CMAKE_BINARY_DIR}/tools/mlir/include )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_subdirectory(tool)

add_clang_library(CIRTidy
  CIRTidy.cpp
  CIRASTConsumer.cpp

  DEPENDS
  omp_gen

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
)

clang_target_link_libraries(CIRTidy
  PRIVATE
  clangBasic
  clangTooling
  clangToolingCore
  )
