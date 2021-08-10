add_flang_unittest(FlangFrontendTests
  CompilerInstanceTest.cpp
  FrontendActionTest.cpp
)

target_link_libraries(FlangFrontendTests
  PRIVATE
  clang-cpp
  flangFrontend
  flangFrontendTool
  FortranParser
  FortranSemantics
  FortranCommon
  FortranEvaluate
)
