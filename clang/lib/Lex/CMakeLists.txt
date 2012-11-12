# TODO: Add -maltivec when ARCH is PowerPC.

set(LLVM_LINK_COMPONENTS support)

add_clang_library(clangLex
  HeaderMap.cpp
  HeaderSearch.cpp
  Lexer.cpp
  LiteralSupport.cpp
  MacroArgs.cpp
  MacroInfo.cpp
  ModuleMap.cpp
  PPCaching.cpp
  PPCallbacks.cpp
  PPDirectives.cpp
  PPExpressions.cpp
  PPLexerChange.cpp
  PPMacroExpansion.cpp
  PTHLexer.cpp
  Pragma.cpp
  PreprocessingRecord.cpp
  Preprocessor.cpp
  PreprocessorLexer.cpp
  ScratchBuffer.cpp
  TokenConcatenation.cpp
  TokenLexer.cpp
  )

add_dependencies(clangLex
  ClangAttrSpellings
  ClangDiagnosticCommon
  ClangDiagnosticLex
  )

target_link_libraries(clangLex
  clangBasic
  )
