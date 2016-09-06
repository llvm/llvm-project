set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_clang_library(clangIndex
  CodegenNameGenerator.cpp
  CommentToXML.cpp
  IndexBody.cpp
  IndexDecl.cpp
  IndexingAction.cpp
  IndexingContext.cpp
  IndexSymbol.cpp
  IndexTypeSourceInfo.cpp
  USRGeneration.cpp

  ADDITIONAL_HEADERS
  IndexingContext.h
  SimpleFormatContext.h

  LINK_LIBS
  clangAST
  clangBasic
  clangFormat
  clangFrontend
  clangRewrite
  clangToolingCore
  )
