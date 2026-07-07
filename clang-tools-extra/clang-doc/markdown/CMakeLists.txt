set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangDocMarkdown STATIC
  Markdown.cpp
  )
