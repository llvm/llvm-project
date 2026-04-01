set(LIBS
  AIIRAnalysis
  AIIRIR
  AIIRParser
  AIIRRegisterAllDialects
  AIIRRegisterAllExtensions
  AIIRRegisterAllPasses
  AIIRSupport
  AIIRTransformDialect
  AIIRTransformDialectTransforms
  AIIRTransforms
)

add_aiir_tool(aiir-transform-opt
  aiir-transform-opt.cpp

  DEPENDS
  ${LIBS}
)
target_link_libraries(aiir-transform-opt PRIVATE ${LIBS})
llvm_update_compile_flags(aiir-transform-opt)
aiir_check_all_link_libraries(aiir-transform-opt)
