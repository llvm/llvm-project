set(LLVM_LINK_COMPONENTS
  Core
  Option
  Support
  TargetParser
  )

set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(ScanDepsOptsTableGen)

add_clang_tool(clang-scan-deps
  ClangScanDeps.cpp

  DEPENDS
  ScanDepsOptsTableGen
  )

set(CLANG_SCAN_DEPS_LIB_DEPS
  clangAST
  clangBasic
  clangCodeGen
  clangDependencyScanning
  clangDriver
  clangFrontend
  clangFrontendTool
  clangLex
  clangParse
  clangSerialization
  clangTooling
  )

clang_target_link_libraries(clang-scan-deps
  PRIVATE
  ${CLANG_SCAN_DEPS_LIB_DEPS}
  )

