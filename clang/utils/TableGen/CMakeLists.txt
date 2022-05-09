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
  ClangSyntaxEmitter.cpp
  ClangTypeNodesEmitter.cpp
  MveEmitter.cpp
  NeonEmitter.cpp
  RISCVVEmitter.cpp
  SveEmitter.cpp
  TableGen.cpp
  )

target_link_libraries(clang-tblgen PRIVATE clangSupport)

set_target_properties(clang-tblgen PROPERTIES FOLDER "Clang tablegenning")
