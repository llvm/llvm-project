set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Core
  FrontendOpenACC
  FrontendOpenMP
  TargetParser
)

add_flang_unittest(OpenMPUtilsTests
  OpenMPUtils.cpp
)

target_link_libraries(OpenMPUtilsTests
  PRIVATE
  flangFrontend
  flangFrontendTool
  FortranLower
  FortranParser
  FortranSemantics
  FortranSupport
  FortranEvaluate
)

clang_target_link_libraries(OpenMPUtilsTests
  PRIVATE
  clangBasic
)

mlir_target_link_libraries(OpenMPUtilsTests
  PRIVATE
  MLIRIR
)
