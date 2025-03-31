get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${test_libs}
  clangCIR
  clangCIRLoweringDirectToLLVM
  MLIRCIR
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIRLspServerLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  )

add_mlir_tool(cir-lsp-server
  cir-lsp-server.cpp

  DEPENDS
  ${LIBS}
)

target_link_libraries(cir-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(cir-lsp-server)
