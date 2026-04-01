include_directories(${LLVM_MAIN_SRC_DIR}/../aiir/include)
include_directories(${CMAKE_BINARY_DIR}/tools/aiir/include)

set(LIBS
  ${test_libs}
  clangCIR
  clangCIRLoweringDirectToLLVM
  AIIRAffineAnalysis
  AIIRAnalysis
  AIIRCIR
  AIIRDialect
  AIIRIR
  AIIRLspServerLib
  AIIRParser
  AIIRPass
  AIIRRegisterAllDialects
  AIIRRegisterAllPasses
  AIIRSupport
  AIIRTransformUtils
  AIIRTransforms
  )

add_aiir_tool(cir-lsp-server
  cir-lsp-server.cpp

  DEPENDS
  ${LIBS}
)

target_link_libraries(cir-lsp-server PRIVATE ${LIBS})
llvm_update_compile_flags(cir-lsp-server)
