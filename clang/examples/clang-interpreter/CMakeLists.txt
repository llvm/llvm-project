set(LLVM_LINK_COMPONENTS
  jit
  interpreter
  nativecodegen
  asmparser
  bitreader
  bitwriter
  codegen
  ipo
  linker
  selectiondag
  )

add_clang_executable(clang-interpreter
  main.cpp
  )

add_dependencies(clang-interpreter
  clang-headers
  )

target_link_libraries(clang-interpreter
  clangFrontend
  clangSerialization
  clangDriver
  clangCodeGen
  clangSema
  clangStaticAnalyzerFrontend
  clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore
  clangAnalysis
  clangRewriteCore
  clangRewriteFrontend
  clangAST
  clangParse
  clangLex
  clangBasic
  )
