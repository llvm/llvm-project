add_aiir_library(AIIRAsmParser
  AffineParser.cpp
  AsmParserState.cpp
  AttributeParser.cpp
  DialectSymbolParser.cpp
  Lexer.cpp
  LocationParser.cpp
  Parser.cpp
  Token.cpp
  TypeParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Parser

  LINK_LIBS PUBLIC
  AIIRIR
  )
