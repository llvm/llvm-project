# clang-doc/support contains support libraries that do not depend
# on clang either programmatically or conceptually.
set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangDocSupport STATIC
  File.cpp
  Utils.cpp
  )
