macro(clang_diag_gen component)
  clang_tablegen(Diagnostic${component}Kinds.inc
    -gen-clang-diags-defs -clang-component=${component}
    SOURCE Diagnostic.td
    TARGET ClangDiagnostic${component})
endmacro(clang_diag_gen)

clang_diag_gen(Analysis)
clang_diag_gen(AST)
clang_diag_gen(Comment)
clang_diag_gen(Common)
clang_diag_gen(CrossTU)
clang_diag_gen(Driver)
clang_diag_gen(Frontend)
clang_diag_gen(Lex)
clang_diag_gen(Parse)
clang_diag_gen(Refactoring)
clang_diag_gen(Sema)
clang_diag_gen(Serialization)
clang_tablegen(DiagnosticGroups.inc -gen-clang-diag-groups
  SOURCE Diagnostic.td
  TARGET ClangDiagnosticGroups)

clang_tablegen(DiagnosticIndexName.inc -gen-clang-diags-index-name
  SOURCE Diagnostic.td
  TARGET ClangDiagnosticIndexName)

clang_tablegen(AttrList.inc -gen-clang-attr-list
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE Attr.td
  TARGET ClangAttrList)

clang_tablegen(AttrSubMatchRulesList.inc -gen-clang-attr-subject-match-rule-list
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE Attr.td
  TARGET ClangAttrSubjectMatchRuleList)

clang_tablegen(AttrHasAttributeImpl.inc -gen-clang-attr-has-attribute-impl
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE Attr.td
  TARGET ClangAttrHasAttributeImpl
  )

# ARM NEON
clang_tablegen(arm_neon.inc -gen-arm-neon-sema
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE arm_neon.td
  TARGET ClangARMNeon)
clang_tablegen(arm_fp16.inc -gen-arm-neon-sema
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE arm_fp16.td
  TARGET ClangARMFP16)
