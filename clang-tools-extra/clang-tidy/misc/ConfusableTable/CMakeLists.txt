set(LLVM_LINK_COMPONENTS Support)
list(REMOVE_ITEM LLVM_COMMON_DEPENDS clang-tablegen-targets)

add_clang_executable(clang-tidy-confusable-chars-gen
  BuildConfusableTable.cpp
  )
