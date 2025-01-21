set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  TargetParser
  Core
)

add_flang_unittest(FlangFrontendTests
  CodeGenActionTest.cpp
  CompilerInstanceTest.cpp
  FrontendActionTest.cpp
)

target_link_libraries(FlangFrontendTests
  PRIVATE
  clangBasic
  flangFrontend
  flangFrontendTool
  FortranLower
  FortranParser
  FortranSemantics
  FortranCommon
  FortranEvaluate
)

mlir_target_link_libraries(FlangFrontendTests
  PRIVATE
  MLIRIR
)
