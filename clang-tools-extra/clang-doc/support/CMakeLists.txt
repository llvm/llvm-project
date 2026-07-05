# clang-doc/support contains support libraries that do not depend
# on clang either programmatically or conceptually.
set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangDocSupport STATIC
  PARTIAL_SOURCES_INTENDED
  File.cpp
  Utils.cpp
  )

add_clang_library(clangDocMarkdown STATIC
  PARTIAL_SOURCES_INTENDED
  Markdown.cpp
  )
  