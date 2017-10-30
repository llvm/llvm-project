set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_unittest(FormatTests
  CleanupTest.cpp
  FormatTest.cpp
  FormatTestComments.cpp
  FormatTestJS.cpp
  FormatTestJava.cpp
  FormatTestObjC.cpp
  FormatTestProto.cpp
  FormatTestRawStrings.cpp
  FormatTestSelective.cpp
  FormatTestTextProto.cpp
  NamespaceEndCommentsFixerTest.cpp
  SortImportsTestJS.cpp
  SortIncludesTest.cpp
  UsingDeclarationsSorterTest.cpp
  )

target_link_libraries(FormatTests
  clangBasic
  clangFormat
  clangFrontend
  clangRewrite
  clangToolingCore
  )
