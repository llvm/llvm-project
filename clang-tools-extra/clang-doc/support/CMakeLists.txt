# clang-doc/support contains support libraries that do not depend
# on clang either programmatically or conceptually.
set(LLVM_LINK_COMPONENTS
  Support
  )

# Utils.cpp references ClangDocContext which is not visible when combined
# with File.cpp in a unity build due to include ordering.
set_source_files_properties(
  Utils.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON
)

add_clang_library(clangDocSupport STATIC
  File.cpp
  Utils.cpp
  )
