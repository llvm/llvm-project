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

  include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
  include_directories(${LLVM_BINARY_DIR}/tools/mlir/include)
endif()

if(CLANG_ENABLE_STATIC_ANALYZER)
  list(APPEND link_libs
    clangStaticAnalyzerFrontend
    )
endif()

if(CLANG_ENABLE_CIR AND ((MINGW OR CYGWIN) AND CLANG_BUILD_CLANG_DYLIB AND NOT BUILD_SHARED_LIBS))
  get_property(save_clang_libs GLOBAL PROPERTY CLANG_STATIC_LIBS)
  get_property(cir_clang_libs GLOBAL PROPERTY CLANG_CIR_STATIC_LIBS)
  set_property(GLOBAL PROPERTY CLANG_STATIC_LIBS ${cir_clang_libs})
endif()
add_clang_library(clangFrontendTool
  ExecuteCompilerInvocation.cpp

  DEPENDS
  ClangDriverOptions
  ${deps}

  LINK_LIBS
  ${link_libs}
  )
if(CLANG_ENABLE_CIR AND ((MINGW OR CYGWIN) AND CLANG_BUILD_CLANG_DYLIB AND NOT BUILD_SHARED_LIBS))
  get_property(cir_clang_libs GLOBAL PROPERTY CLANG_STATIC_LIBS)
  set_property(GLOBAL PROPERTY CLANG_STATIC_LIBS ${save_clang_libs})
  set_property(GLOBAL PROPERTY CLANG_CIR_STATIC_LIBS ${cir_clang_libs})
endif()

if(CLANG_ENABLE_CIR)
  target_include_directories(clangFrontendTool PRIVATE ${LLVM_MAIN_SRC_DIR}/../mlir/include)
  target_include_directories(clangFrontendTool PRIVATE ${LLVM_BINARY_DIR}/tools/mlir/include)
endif()
