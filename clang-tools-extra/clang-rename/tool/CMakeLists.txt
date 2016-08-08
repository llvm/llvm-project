add_clang_executable(clang-rename ClangRename.cpp)

target_link_libraries(clang-rename
  clangBasic
  clangFrontend
  clangRename
  clangRewrite
  clangTooling
  clangToolingCore
  )

install(TARGETS clang-rename RUNTIME DESTINATION bin)

install(PROGRAMS clang-rename.py
  DESTINATION share/clang
  COMPONENT clang-rename)
install(PROGRAMS clang-rename.el
  DESTINATION share/clang
  COMPONENT clang-rename)
