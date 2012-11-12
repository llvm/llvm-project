set(LLVM_LINK_COMPONENTS Support)

add_tablegen(clang-tblgen CLANG
  ClangASTNodesEmitter.cpp
  ClangAttrEmitter.cpp
  ClangCommentCommandInfoEmitter.cpp
  ClangCommentHTMLTagsEmitter.cpp
  ClangDiagnosticsEmitter.cpp
  ClangSACheckersEmitter.cpp
  NeonEmitter.cpp
  OptParserEmitter.cpp
  TableGen.cpp
  )
