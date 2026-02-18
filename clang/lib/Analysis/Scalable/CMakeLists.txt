set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangAnalysisScalable
  ASTEntityMapping.cpp
  Model/BuildNamespace.cpp
  Model/EntityIdTable.cpp
  Model/EntityName.cpp
  Serialization/JSONFormat.cpp
  Serialization/SerializationFormatRegistry.cpp
  TUSummary/ExtractorRegistry.cpp
<<<<<<< HEAD
  Analyses/UnsafeBufferUsage.cpp
=======
  TUSummary/TUSummaryBuilder.cpp
>>>>>>> eng/PR-170176278

  LINK_LIBS
  clangAST
  clangASTMatchers
  clangBasic
  clangIndex
  clangLex
  clangFrontend

  DEPENDS
  )
