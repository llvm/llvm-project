set(LLVM_LINK_COMPONENTS support)

set(LLVM_COMMON_DEPENDS_OLD ${LLVM_COMMON_DEPENDS})

# Drop clang-tablegen-targets from LLVM_COMMON_DEPENDS.
# so that we could use clangToolingCoreNoClang within clang-tblgen.
list(REMOVE_ITEM LLVM_COMMON_DEPENDS clang-tablegen-targets)

add_clang_library(clangToolingCoreNoClang
  Diagnostic.cpp
  Replacement.cpp

  PARTIAL_SOURCES_INTENDED
  )

set(LLVM_COMMON_DEPENDS ${LLVM_COMMON_DEPENDS_OLD})

add_clang_library(clangToolingCore
  DiagnosticClang.cpp
  ReplacementClang.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS
  clangBasic
  clangLex
  clangRewrite
  clangToolingCoreNoClang
  )
