set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

set(link_libs
  clangScalableStaticAnalysisAnalyses
  clangScalableStaticAnalysisCore
  clangScalableStaticAnalysisFrontend
  clangScalableStaticAnalysisSourceTransformation
  clangBasic
  clangCodeGen
  clangDriver
  clangOptions
  clangExtractAPI
  clangFrontend
  clangRewriteFrontend
  )

set(deps)

if(CLANG_ENABLE_CIR)
  list(APPEND deps
    MLIRCIRPassIncGen
    mlir-headers
    )
  list(APPEND link_libs
    clangCIRFrontendAction
    MLIRCIRTransforms
    MLIRIR
    MLIRPass
    )

  include_directories(${CLANG_CIR_MLIR_INCLUDE_DIRS})
endif()

if(CLANG_ENABLE_STATIC_ANALYZER)
  list(APPEND link_libs
    clangStaticAnalyzerFrontend
    )
endif()

add_clang_library(clangFrontendTool
  ExecuteCompilerInvocation.cpp

  DEPENDS
  ClangDriverOptions
  ${deps}

  LINK_LIBS
  ${link_libs}
  )

if(CLANG_ENABLE_CIR)
  target_include_directories(clangFrontendTool PRIVATE ${CLANG_CIR_MLIR_INCLUDE_DIRS})
endif()
