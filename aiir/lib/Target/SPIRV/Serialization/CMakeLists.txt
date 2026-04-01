add_aiir_translation_library(AIIRSPIRVSerialization
  Serialization.cpp
  Serializer.cpp
  SerializeOps.cpp

  DEPENDS
  AIIRSPIRVSerializationGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRSPIRVDialect
  AIIRSPIRVBinaryUtils
  AIIRSupport
  AIIRTranslateLib
  )
