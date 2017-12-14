set(LLVM_LINK_COMPONENTS
  support
  )

get_filename_component(CLANGD_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clangd REALPATH)
include_directories(
  ${CLANGD_SOURCE_DIR}
  )

add_extra_unittest(ClangdTests
  ClangdTests.cpp
  CodeCompleteTests.cpp
  ContextTests.cpp
  FuzzyMatchTests.cpp
  IndexTests.cpp
  JSONExprTests.cpp
  TestFS.cpp
  TraceTests.cpp
  SymbolCollectorTests.cpp
  )

target_link_libraries(ClangdTests
  PRIVATE
  clangBasic
  clangDaemon
  clangFormat
  clangFrontend
  clangIndex
  clangSema
  clangTooling
  clangToolingCore
  LLVMSupport
  )
