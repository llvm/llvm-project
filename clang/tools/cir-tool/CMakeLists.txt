get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  clangCIR
  MLIRAnalysis
  MLIRCIR
  MLIRCIRTransforms
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

add_clang_tool(cir-tool
  cir-tool.cpp

  DEPENDS
  ${LIBS}
)

target_link_libraries(cir-tool PRIVATE ${LIBS})
llvm_update_compile_flags(cir-tool)
