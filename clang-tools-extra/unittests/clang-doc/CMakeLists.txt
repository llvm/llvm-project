set(LLVM_LINK_COMPONENTS
  support
  BitReader
  BitWriter
  )

get_filename_component(CLANG_DOC_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clang-doc REALPATH)
include_directories(
  ${CLANG_DOC_SOURCE_DIR}
  )

add_extra_unittest(ClangDocTests
  BitcodeTest.cpp
  ClangDocTest.cpp
  MDGeneratorTest.cpp
  MergeTest.cpp
  SerializeTest.cpp
  YAMLGeneratorTest.cpp
  )

target_link_libraries(ClangDocTests
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangDoc
  clangFormat
  clangFrontend
  clangRewrite
  clangSerialization
  clangTooling
  clangToolingCore
  )
