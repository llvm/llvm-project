set(LLVM_LINK_COMPONENTS
  Option
  Support
  )

set(link_libs
  clangBasic
  clangCodeGen
  clangDriver
  clangExtractAPI
  clangFrontend
  clangRewriteFrontend
  )

set(deps)

if(CLANG_ENABLE_CIR)
  list(APPEND deps
    MLIRBuiltinOpsIncGen
  )
  list(APPEND link_libs
    clangCIRFrontendAction
    MLIRCIRTransforms
    MLIRIR
    MLIRPass
    )
  list(APPEND deps
    MLIRBuiltinLocationAttributesIncGen
    MLIRBuiltinTypeInterfacesIncGen
    )

  include_directories(${LLVM_MAIN_SRC_DIR}/../mlir/include)
  include_directories(${CMAKE_BINARY_DIR}/tools/mlir/include)
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
  target_include_directories(clangFrontendTool PRIVATE ${LLVM_MAIN_SRC_DIR}/../mlir/include)
  target_include_directories(clangFrontendTool PRIVATE ${CMAKE_BINARY_DIR}/tools/mlir/include)
endif()
