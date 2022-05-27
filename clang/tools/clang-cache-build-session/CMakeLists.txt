set(LLVM_LINK_COMPONENTS
  Support
)

add_clang_tool(clang-cache-build-session
  ClangCacheBuildSession.cpp
  CMakeFileAPI.cpp
)
