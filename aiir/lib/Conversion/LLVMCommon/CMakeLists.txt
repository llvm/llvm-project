add_aiir_conversion_library(AIIRLLVMCommonConversion
  ConversionTarget.cpp
  LoweringOptions.cpp
  MemRefBuilder.cpp
  Pattern.cpp
  PrintCallHelper.cpp
  StructBuilder.cpp
  TypeConverter.cpp
  VectorPattern.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMDialect
  AIIRSupport
  AIIRTransforms
  )
