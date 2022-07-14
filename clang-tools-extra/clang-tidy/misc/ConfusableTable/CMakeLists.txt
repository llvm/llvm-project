set(LLVM_LINK_COMPONENTS Support)
list(REMOVE_ITEM LLVM_COMMON_DEPENDS clang-tablegen-targets)

add_llvm_executable(make-confusable-table
  BuildConfusableTable.cpp
  )
