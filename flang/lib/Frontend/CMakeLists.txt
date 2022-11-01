get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_flang_library(flangFrontend
  CompilerInstance.cpp
  CompilerInvocation.cpp
  CodeGenOptions.cpp
  FrontendAction.cpp
  FrontendActions.cpp
  FrontendOptions.cpp
  TextDiagnosticPrinter.cpp
  TextDiagnosticBuffer.cpp
  TextDiagnostic.cpp

  DEPENDS
  FIRBuilder
  FIRDialect
  FIRSupport
  FIROptTransformsPassIncGen
  HLFIRDialect
  MLIRIR
  ${dialect_libs}

  LINK_LIBS
  FortranParser
  FortranSemantics
  FortranEvaluate
  FortranCommon
  FortranLower
  FIRDialect
  FIRSupport
  FIRBuilder
  FIRCodeGen
  FIRTransforms
  HLFIRDialect
  MLIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRSCFToControlFlow
  ${dialect_libs}

  LINK_COMPONENTS
  Passes
  Analysis
  IRReader
  Option
  Support
  Target
  FrontendOpenACC
  FrontendOpenMP
)

if(CLANG_LINK_CLANG_DYLIB)
  add_dependencies(flangFrontend clang-cpp)
else()
  add_dependencies(flangFrontend clangBasic)
endif()

clang_target_link_libraries(flangFrontend
  PRIVATE
  clangBasic
  clangDriver
)
