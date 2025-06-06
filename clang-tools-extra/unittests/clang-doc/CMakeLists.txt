set(LLVM_LINK_COMPONENTS
  support
  BitstreamReader
  FrontendOpenMP
  )

# Unittests need access to mustache template files, so we use a config file to
# inject those into a config.h header that can provide it to the unittests.
set(CLANG_DOC_TEST_ASSET_DIR "${LLVM_RUNTIME_OUTPUT_INTDIR}/../share/clang-doc")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/config.h.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/config.h"
)

# The config.h file is in ${CMAKE_CURRENT_BINARY_DIR}, so add it to
# include_directories.
get_filename_component(CLANG_DOC_SOURCE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../../clang-doc REALPATH)
include_directories(
  ${CLANG_DOC_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  )

add_extra_unittest(ClangDocTests
  BitcodeTest.cpp
  ClangDocTest.cpp
  GeneratorTest.cpp
  HTMLGeneratorTest.cpp
  HTMLMustacheGeneratorTest.cpp
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
  LLVMTestingSupport
  )
