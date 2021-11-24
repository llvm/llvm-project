add_clang_tool(cir-tool cir-tool.cpp)
llvm_update_compile_flags(cir-tool)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

include_directories( ${LLVM_MAIN_SRC_DIR}/../mlir/include )
include_directories( ${CMAKE_BINARY_DIR}/tools/mlir/include )

target_link_libraries(cir-tool PRIVATE
  ${dialect_libs}
  ${conversion_libs}

  clangCIR
  MLIRAnalysis
  MLIRCIR
  MLIRDialect
  MLIRIR
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRTransformUtils
)
