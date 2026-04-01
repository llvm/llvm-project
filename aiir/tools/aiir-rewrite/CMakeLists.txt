set(LLVM_LINK_COMPONENTS
  Support
  )

set(LIBS
  AIIRAffineAnalysis
  AIIRAnalysis
  AIIRCastInterfaces
  AIIRDialect
  AIIRIR
  AIIRParser
  AIIRPass
  AIIRRegisterAllDialects
  AIIRSupport
  AIIRTransformUtils
  AIIRTransforms
  )

include_directories(../../../clang/include)

add_aiir_tool(aiir-rewrite
  aiir-rewrite.cpp

  SUPPORT_PLUGINS
  )
aiir_target_link_libraries(aiir-rewrite PRIVATE ${LIBS})
llvm_update_compile_flags(aiir-rewrite)

aiir_check_all_link_libraries(aiir-rewrite)
export_executable_symbols_for_plugins(aiir-rewrite)
