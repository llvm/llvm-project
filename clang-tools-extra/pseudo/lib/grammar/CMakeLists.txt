set(LLVM_LINK_COMPONENTS Support)

# This library is used by the clang-pseudo-gen tool which runs at build time.
# Dependencies should be minimal to avoid long dep paths in the build graph.
# It does use clangBasic headers (tok::TokenKind), but linking is not needed.
# We have no transitive dependencies on tablegen files.
list(REMOVE_ITEM LLVM_COMMON_DEPENDS clang-tablegen-targets)
add_clang_library(clangPseudoGrammar
  Grammar.cpp
  GrammarBNF.cpp
  LRGraph.cpp
  LRTable.cpp
  LRTableBuild.cpp
  )

