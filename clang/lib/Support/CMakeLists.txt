set(LLVM_COMMON_DEPENDS_OLD ${LLVM_COMMON_DEPENDS})

# Drop clang-tablegen-targets from LLVM_COMMON_DEPENDS.
# so that we could use clangSupport within clang-tblgen and other clang
# component.
list(REMOVE_ITEM LLVM_COMMON_DEPENDS clang-tablegen-targets)

set(LLVM_LINK_COMPONENTS
  Support
  )

add_clang_library(clangSupport
  RISCVVIntrinsicUtils.cpp
  )

set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS_OLD})
