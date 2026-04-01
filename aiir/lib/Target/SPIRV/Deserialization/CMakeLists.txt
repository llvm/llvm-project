add_aiir_translation_library(AIIRSPIRVDeserialization
  DeserializeOps.cpp
  Deserializer.cpp
  Deserialization.cpp

  DEPENDS
  AIIRSPIRVSerializationGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRSPIRVDialect
  AIIRSPIRVBinaryUtils
  AIIRSupport
  AIIRTranslateLib
  )


