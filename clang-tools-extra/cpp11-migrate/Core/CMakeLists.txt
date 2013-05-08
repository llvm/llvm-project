set(LLVM_LINK_COMPONENTS support)

add_clang_library(migrateCore
  Transforms.cpp
  Transform.cpp
  IncludeExcludeInfo.cpp
  )
target_link_libraries(migrateCore
  clangTooling
  clangBasic
  clangASTMatchers
  )
