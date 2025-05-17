set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_clang_library(clangASTImporter
  ASTImporter.cpp
  ASTImporterLookupTable.cpp
  ExternalASTMerger.cpp

  LINK_LIBS
  clangAST
  clangBasic

  DEPENDS
  ClangAttrDocTable
  Opcodes
  )
