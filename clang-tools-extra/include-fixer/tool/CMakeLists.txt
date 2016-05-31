include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_clang_executable(clang-include-fixer ClangIncludeFixer.cpp)
target_link_libraries(clang-include-fixer
  clangBasic
  clangFormat
  clangFrontend
  clangIncludeFixer
  clangRewrite
  clangTooling
  clangToolingCore
  findAllSymbols
  )
