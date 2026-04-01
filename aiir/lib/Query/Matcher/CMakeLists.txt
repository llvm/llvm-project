add_aiir_library(AIIRQueryMatcher
  MatchFinder.cpp
  MatchersInternal.cpp
  Parser.cpp
  RegistryManager.cpp
  VariantValue.cpp
  Diagnostics.cpp
  ErrorBuilder.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Query/Matcher

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRParser
  )
