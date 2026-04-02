set(LLVM_LINK_COMPONENTS
  Support
  TargetParser
  )

# SymbolGraphSerializer.cpp uses "using namespace llvm" which brings types
# that conflict with clang::extractapi types in unity builds.
set_source_files_properties(
  Serialization/SymbolGraphSerializer.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
)

add_clang_library(clangExtractAPI
  API.cpp
  APIIgnoresList.cpp
  ExtractAPIConsumer.cpp
  DeclarationFragments.cpp
  Serialization/SymbolGraphSerializer.cpp
  TypedefUnderlyingTypeResolver.cpp

  LINK_LIBS
  clangAST
  clangBasic
  clangFrontend
  clangInstallAPI
  clangLex
  clangUnifiedSymbolResolution
  )
