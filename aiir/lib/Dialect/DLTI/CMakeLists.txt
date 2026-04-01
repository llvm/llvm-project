add_subdirectory(TransformOps)
add_aiir_dialect_library(AIIRDLTIDialect
  DLTI.cpp
  Traits.cpp

  DEPENDS
  AIIRDLTIIncGen

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRDataLayoutInterfaces
  )
