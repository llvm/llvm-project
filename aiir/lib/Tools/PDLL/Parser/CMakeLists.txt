
llvm_add_library(AIIRPDLLParser STATIC
  CodeComplete.cpp
  Lexer.cpp
  Parser.cpp
  
  DISABLE_LLVM_LINK_LLVM_DYLIB

  LINK_COMPONENTS
  Support
  TableGen

  LINK_LIBS PUBLIC
  AIIRPDLLAST
  AIIRSupport
  AIIRTableGen
  )
