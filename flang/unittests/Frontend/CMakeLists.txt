add_flang_unittest(FlangFrontendTests
  CompilerInstanceTest.cpp
  InputOutputTest.cpp
)

target_link_libraries(FlangFrontendTests
  PRIVATE
  clangBasic
  clangFrontend
  flangFrontend
  flangFrontendTool
  FortranParser
  )
