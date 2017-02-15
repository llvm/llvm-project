set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(FormatTests
  CleanupTest.cpp
  FormatTest.cpp
  FormatTestJava.cpp
  FormatTestJS.cpp
  FormatTestProto.cpp
  FormatTestSelective.cpp
  SortImportsTestJS.cpp
  SortIncludesTest.cpp
  )

target_link_libraries(FormatTests
  clangBasic
  clangFormat
  clangFrontend
  clangRewrite
  clangToolingCore
  )
