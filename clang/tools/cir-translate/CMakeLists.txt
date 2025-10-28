get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)

add_clang_tool(cir-translate
  cir-translate.cpp
)

clang_target_link_libraries(cir-translate
  PRIVATE
  clangCIR
  clangCIRLoweringDirectToLLVM
  MLIRCIR
  MLIRCIRTransforms
)

target_link_libraries(cir-translate
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${translation_libs}
  MLIRAnalysis
  MLIRDialect
  MLIRIR
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRTranslateLib
  MLIRSupport
)
