include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_clang_tool(clang-include-fixer
  ClangIncludeFixer.cpp
  )

clang_target_link_libraries(clang-include-fixer
  PRIVATE
  clangBasic
  clangFormat
  clangFrontend
  clangRewrite
  clangSerialization
  clangTooling
  clangToolingCore
  )
target_link_libraries(clang-include-fixer
  PRIVATE
  clangIncludeFixer
  findAllSymbols
  )

install(PROGRAMS clang-include-fixer.el
  DESTINATION share/clang
  COMPONENT clang-include-fixer)
install(PROGRAMS clang-include-fixer.py
  DESTINATION share/clang
  COMPONENT clang-include-fixer)
