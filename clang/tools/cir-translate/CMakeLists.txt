get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY AIIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY AIIR_TRANSLATION_LIBS)

include_directories(${LLVM_MAIN_SRC_DIR}/../aiir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/aiir/include)

add_clang_tool(cir-translate
  cir-translate.cpp
)

clang_target_link_libraries(cir-translate
  PRIVATE
  clangCIR
  clangCIRLoweringDirectToLLVM
  AIIRCIR
  AIIRCIRTransforms
)

target_link_libraries(cir-translate
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${translation_libs}
  AIIRAnalysis
  AIIRDialect
  AIIRIR
  AIIROptLib
  AIIRParser
  AIIRPass
  AIIRTransforms
  AIIRTransformUtils
  AIIRTranslateLib
  AIIRSupport
)
