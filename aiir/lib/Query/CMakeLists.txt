add_aiir_library(AIIRQuery
  Query.cpp
  QueryParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Query

  LINK_LIBS PUBLIC
  AIIRFuncDialect
  AIIRQueryMatcher
  )

add_subdirectory(Matcher)
