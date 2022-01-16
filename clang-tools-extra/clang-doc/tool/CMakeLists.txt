include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_clang_tool(clang-doc
  ClangDocMain.cpp
  )

clang_target_link_libraries(clang-doc
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangFrontend
  clangTooling
  clangToolingCore
  )
target_link_libraries(clang-doc
  PRIVATE
  clangDoc
  )

install(FILES ../assets/clang-doc-default-stylesheet.css
  DESTINATION share/clang
  COMPONENT clang-doc)

install(FILES ../assets/index.js
  DESTINATION share/clang
  COMPONENT clang-doc)
