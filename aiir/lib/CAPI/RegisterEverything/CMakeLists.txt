# Dialect registration.
get_property(translation_libs GLOBAL PROPERTY AIIR_TRANSLATION_LIBS)
add_aiir_upstream_c_api_library(AIIRCAPIRegisterEverything
  RegisterEverything.cpp

  LINK_LIBS PUBLIC
  ${translation_libs}

  AIIRBuiltinToLLVMIRTranslation
  AIIRCAPIIR
  AIIRCAPITransforms
  AIIRLLVMToLLVMIRTranslation
  AIIRRegisterAllDialects
  AIIRRegisterAllExtensions
  AIIRRegisterAllPasses
)
