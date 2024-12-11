add_clang_library(MLIRCIR
  CIRAttrs.cpp
  CIRDialect.cpp
  CIRTypes.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  )
