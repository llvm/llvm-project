add_flang_unittest(FlangFrontendTests
  CompilerInstanceTest.cpp
)

target_link_libraries(FlangFrontendTests
  PRIVATE
  LLVMSupport
  clangBasic
  flangFrontend
  flangFrontendTool)
