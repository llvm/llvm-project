set(LLVM_LINK_COMPONENTS
  support
  BitstreamReader
  FrontendOpenMP
  )

get_filename_component(CLANG_DOC_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clang-doc REALPATH)
include_directories(
  ${CLANG_DOC_SOURCE_DIR}
  )

add_extra_unittest(ClangDocTests
  BitcodeTest.cpp
  ClangDocTest.cpp
  GeneratorTest.cpp
  HTMLGeneratorTest.cpp
  MDGeneratorTest.cpp
  MergeTest.cpp
  SerializeTest.cpp
  YAMLGeneratorTest.cpp
  )

clang_target_link_libraries(ClangDocTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFormat
  clangFrontend
  clangRewrite
  clangSerialization
  clangTooling
  clangToolingCore
  )
target_link_libraries(ClangDocTests
  PRIVATE
  clangDoc
  )
