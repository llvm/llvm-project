set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_clang_library(clangSummary
  SummaryAttribute.cpp
  SummaryConsumer.cpp
  SummaryContext.cpp

  LINK_LIBS
  clangAST
  clangSupport
  )
