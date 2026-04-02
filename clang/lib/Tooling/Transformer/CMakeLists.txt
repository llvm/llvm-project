set(LLVM_LINK_COMPONENTS
  FrontendOpenMP
  Support
)

# Both files define a static getNode() helper in anonymous namespaces,
# causing redefinition errors in unity builds.
set_source_files_properties(
  Stencil.cpp
  RangeSelector.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
)

add_clang_library(clangTransformer
  Parsing.cpp
  RangeSelector.cpp
  RewriteRule.cpp
  SourceCode.cpp
  SourceCodeBuilders.cpp
  Stencil.cpp
  Transformer.cpp

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangLex
  clangToolingCore
  clangToolingRefactoring

  DEPENDS
  omp_gen
  )
