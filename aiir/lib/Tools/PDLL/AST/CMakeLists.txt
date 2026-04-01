add_aiir_library(AIIRPDLLAST
  Context.cpp
  Diagnostic.cpp
  NodePrinter.cpp
  Nodes.cpp
  Types.cpp

  LINK_LIBS PUBLIC
  AIIRPDLLODS
  AIIRSupport
  )
