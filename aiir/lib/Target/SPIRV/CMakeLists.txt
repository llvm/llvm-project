add_subdirectory(Deserialization)
add_subdirectory(Serialization)

set(LLVM_OPTIONAL_SOURCES
  SPIRVBinaryUtils.cpp
  TranslateRegistration.cpp
  Target.cpp
  )

add_aiir_translation_library(AIIRSPIRVBinaryUtils
  SPIRVBinaryUtils.cpp

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRSupport
  )

add_aiir_translation_library(AIIRSPIRVTranslateRegistration
  TranslateRegistration.cpp

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRSPIRVSerialization
  AIIRSPIRVDeserialization
  AIIRSupport
  )

add_aiir_dialect_library(AIIRSPIRVTarget
  Target.cpp

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRGPUDialect
  AIIRSPIRVSerialization
  AIIRSupport
  )
