set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsInfos
  FrontendOpenMP
  support
  )

# Needed by LLVM's CMake checks because this file defines multiple targets.
set(LLVM_OPTIONAL_SOURCES ClangTidyMain.cpp ClangTidyToolMain.cpp)

add_clang_library(clangTidyMain
  ClangTidyMain.cpp

  LINK_LIBS
  clangTidy
  ${ALL_CLANG_TIDY_CHECKS}
  )

clang_target_link_libraries(clangTidyMain
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangTooling
  clangToolingCore
  )

add_clang_tool(clang-tidy
  ClangTidyToolMain.cpp
  )
add_dependencies(clang-tidy
  clang-resource-headers
  )
clang_target_link_libraries(clang-tidy
  PRIVATE
  clangAST
  clangASTMatchers
  clangBasic
  clangTooling
  clangToolingCore
  )
target_link_libraries(clang-tidy
  PRIVATE
  clangTidy
  clangTidyMain
  ${ALL_CLANG_TIDY_CHECKS}
  )


install(PROGRAMS clang-tidy-diff.py
  DESTINATION share/clang
  COMPONENT clang-tidy)
install(PROGRAMS run-clang-tidy.py
  DESTINATION share/clang
  COMPONENT clang-tidy)
