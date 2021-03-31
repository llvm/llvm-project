add_flang_unittest(FlangFrontendTests
  CompilerInstanceTest.cpp
  FrontendActionTest.cpp
)

target_link_libraries(FlangFrontendTests
  PRIVATE
  clangBasic
  flangFrontend
  flangFrontendTool
  FortranParser
  FortranSemantics
  FortranCommon
)
