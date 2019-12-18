set(LLVM_LINK_COMPONENTS Support)

add_tablegen(clang-tblgen CLANG
  ASTTableGen.cpp
  ClangASTNodesEmitter.cpp
  ClangASTPropertiesEmitter.cpp
  ClangAttrEmitter.cpp
  ClangCommentCommandInfoEmitter.cpp
  ClangCommentHTMLNamedCharacterReferenceEmitter.cpp
  ClangCommentHTMLTagsEmitter.cpp
  ClangDataCollectorsEmitter.cpp
  ClangDiagnosticsEmitter.cpp
  ClangOpcodesEmitter.cpp
  ClangOpenCLBuiltinEmitter.cpp
  ClangOptionDocEmitter.cpp
  ClangSACheckersEmitter.cpp
  ClangTypeNodesEmitter.cpp
  MveEmitter.cpp
  NeonEmitter.cpp
  TableGen.cpp
  )
set_target_properties(clang-tblgen PROPERTIES FOLDER "Clang tablegenning")
