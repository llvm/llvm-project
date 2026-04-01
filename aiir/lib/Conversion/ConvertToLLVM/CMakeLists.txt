set(LLVM_OPTIONAL_SOURCES
  ConvertToLLVMPass.cpp
  ToLLVMInterface.cpp
)

add_aiir_conversion_library(AIIRConvertToLLVMInterface
  ToLLVMInterface.cpp

  DEPENDS
  AIIRConvertToLLVMInterfaceIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRSupport
)

add_aiir_conversion_library(AIIRConvertToLLVMPass
  ConvertToLLVMPass.cpp

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRConvertToLLVMInterface
  AIIRLLVMCommonConversion
  AIIRLLVMDialect
  AIIRPass
  AIIRRewrite
  AIIRSupport
  AIIRTransformUtils
  )
