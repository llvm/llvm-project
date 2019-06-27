set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_clang_tool(clang-scan-deps
  ClangScanDeps.cpp
  )

set(CLANG_SCAN_DEPS_LIB_DEPS
  clangAST
  clangBasic
  clangCodeGen
  clangDriver
  clangFrontend
  clangFrontendTool
  clangLex
  clangParse
  clangSerialization
  clangTooling
  )

target_link_libraries(clang-scan-deps
  PRIVATE
  ${CLANG_SCAN_DEPS_LIB_DEPS}
  )

