# TODO: Add -maltivec when ARCH is PowerPC.

set(LLVM_LINK_COMPONENTS
  Support
  TargetParser
  )

# "using namespace clang::dependency_directives_scan" at file scope brings
# dependency_directives_scan::Token into scope, conflicting with clang::Token.
set_source_files_properties(
  DependencyDirectivesScanner.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
)

add_clang_library(clangLex
  DependencyDirectivesScanner.cpp
  HeaderMap.cpp
  HeaderSearch.cpp
  InitHeaderSearch.cpp
  Lexer.cpp
  LexHLSLRootSignature.cpp
  LiteralSupport.cpp
  MacroArgs.cpp
  MacroInfo.cpp
  ModuleMap.cpp
  ModuleMapFile.cpp
  PPCaching.cpp
  PPCallbacks.cpp
  PPConditionalDirectiveRecord.cpp
  PPDirectives.cpp
  PPExpressions.cpp
  PPLexerChange.cpp
  PPMacroExpansion.cpp
  Pragma.cpp
  PreprocessingRecord.cpp
  Preprocessor.cpp
  PreprocessorLexer.cpp
  ScratchBuffer.cpp
  TokenConcatenation.cpp
  TokenLexer.cpp

  LINK_LIBS
  clangBasic
  )
