macro(clang_diag_gen component)
  clang_tablegen(Diagnostic${component}Kinds.inc
    -gen-clang-diags-defs -clang-component=${component}
    SOURCE Diagnostic.td
    TARGET ClangDiagnostic${component})

  clang_tablegen(Diagnostic${component}Enums.inc
    -gen-clang-diags-enums -clang-component=${component}
    SOURCE Diagnostic.td
    TARGET ClangDiagnostic${component}Enums)

  clang_tablegen(Diagnostic${component}CompatIDs.inc
    -gen-clang-diags-compat-ids -clang-component=${component}
    SOURCE Diagnostic.td
    TARGET ClangDiagnostic${component}CompatIDs)

  clang_tablegen(Diagnostic${component}Interface.inc
    -gen-clang-diags-iface -clang-component=${component}
    SOURCE Diagnostic.td
    TARGET ClangDiagnostic${component}Interface)
endmacro(clang_diag_gen)

clang_diag_gen(Analysis)
clang_diag_gen(AST)
clang_diag_gen(Comment)
clang_diag_gen(Common)
clang_diag_gen(CrossTU)
clang_diag_gen(Driver)
clang_diag_gen(Frontend)
clang_diag_gen(InstallAPI)
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

clang_tablegen(DiagnosticAllCompatIDs.inc
  -gen-clang-diags-compat-ids
  SOURCE Diagnostic.td
  TARGET ClangDiagnosticAllCompatIDs)

clang_tablegen(AttrList.inc -gen-clang-attr-list
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE Attr.td
  TARGET ClangAttrList)

clang_tablegen(AttrParsedAttrList.inc -gen-clang-attr-parsed-attr-list
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE Attr.td
  TARGET ClangAttrParsedAttrList)

clang_tablegen(AttrSubMatchRulesList.inc -gen-clang-attr-subject-match-rule-list
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE Attr.td
  TARGET ClangAttrSubjectMatchRuleList)

clang_tablegen(RegularKeywordAttrInfo.inc -gen-clang-regular-keyword-attr-info
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE Attr.td
  TARGET ClangRegularKeywordAttrInfo
  )

clang_tablegen(AttrHasAttributeImpl.inc -gen-clang-attr-has-attribute-impl
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE Attr.td
  TARGET ClangAttrHasAttributeImpl
  )

clang_tablegen(CXX11AttributeInfo.inc -gen-cxx11-attribute-info
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE Attr.td
  TARGET CXX11AttributeInfo
  )

  clang_tablegen(AttributeSpellingList.inc -gen-attribute-spelling-list
  -I ${CMAKE_CURRENT_SOURCE_DIR}/../../
  SOURCE Attr.td
  TARGET AttributeSpellingList
  )

clang_tablegen(Builtins.inc -gen-clang-builtins
  SOURCE Builtins.td
  TARGET ClangBuiltins)

clang_tablegen(BuiltinsBPF.inc -gen-clang-builtins
  SOURCE BuiltinsBPF.td
  TARGET ClangBuiltinsBPF)

clang_tablegen(BuiltinsDirectX.inc -gen-clang-builtins
  SOURCE BuiltinsDirectX.td
  TARGET ClangBuiltinsDirectX)

clang_tablegen(BuiltinsHexagon.inc -gen-clang-builtins
  SOURCE BuiltinsHexagon.td
  TARGET ClangBuiltinsHexagon)

clang_tablegen(BuiltinsNVPTX.inc -gen-clang-builtins
  SOURCE BuiltinsNVPTX.td
  TARGET ClangBuiltinsNVPTX)

clang_tablegen(BuiltinsRISCV.inc -gen-clang-builtins
  SOURCE BuiltinsRISCV.td
  TARGET ClangBuiltinsRISCV)

clang_tablegen(BuiltinsSPIRV.inc -gen-clang-builtins
  SOURCE BuiltinsSPIRV.td
  TARGET ClangBuiltinsSPIRV)

clang_tablegen(BuiltinsX86.inc -gen-clang-builtins
  SOURCE BuiltinsX86.td
  TARGET ClangBuiltinsX86)

clang_tablegen(BuiltinsX86_64.inc -gen-clang-builtins
  SOURCE BuiltinsX86_64.td
  TARGET ClangBuiltinsX86_64)

clang_tablegen(BuiltinTemplates.inc -gen-clang-builtin-templates
  SOURCE BuiltinTemplates.td
  TARGET ClangBuiltinTemplates)

# ARM NEON and MVE
clang_tablegen(arm_neon.inc -gen-arm-neon-sema
  SOURCE arm_neon.td
  TARGET ClangARMNeon)
clang_tablegen(arm_fp16.inc -gen-arm-neon-sema
  SOURCE arm_fp16.td
  TARGET ClangARMFP16)
clang_tablegen(arm_immcheck_types.inc -gen-arm-immcheck-types
  SOURCE arm_sve.td
  TARGET ClangARMImmChecks)
clang_tablegen(arm_mve_builtins.inc -gen-arm-mve-builtin-def
  SOURCE arm_mve.td
  TARGET ClangARMMveBuiltinsDef)
clang_tablegen(arm_mve_builtin_cg.inc -gen-arm-mve-builtin-codegen
  SOURCE arm_mve.td
  TARGET ClangARMMveBuiltinCG)
clang_tablegen(arm_mve_builtin_sema.inc -gen-arm-mve-builtin-sema
  SOURCE arm_mve.td
  TARGET ClangARMMveBuiltinSema)
clang_tablegen(arm_mve_builtin_aliases.inc -gen-arm-mve-builtin-aliases
  SOURCE arm_mve.td
  TARGET ClangARMMveBuiltinAliases)
clang_tablegen(arm_sve_builtins.inc -gen-arm-sve-builtins
  SOURCE arm_sve.td
  TARGET ClangARMSveBuiltins)
clang_tablegen(arm_sve_builtin_cg.inc -gen-arm-sve-builtin-codegen
  SOURCE arm_sve.td
  TARGET ClangARMSveBuiltinCG)
clang_tablegen(arm_sve_typeflags.inc -gen-arm-sve-typeflags
  SOURCE arm_sve.td
  TARGET ClangARMSveTypeFlags)
clang_tablegen(arm_sve_sema_rangechecks.inc -gen-arm-sve-sema-rangechecks
  SOURCE arm_sve.td
  TARGET ClangARMSveSemaRangeChecks)
clang_tablegen(arm_sve_streaming_attrs.inc -gen-arm-sve-streaming-attrs
  SOURCE arm_sve.td
  TARGET ClangARMSveStreamingAttrs)
clang_tablegen(arm_sme_builtins.inc -gen-arm-sme-builtins
  SOURCE arm_sme.td
  TARGET ClangARMSmeBuiltins)
clang_tablegen(arm_sme_builtin_cg.inc -gen-arm-sme-builtin-codegen
  SOURCE arm_sme.td
  TARGET ClangARMSmeBuiltinCG)
clang_tablegen(arm_sme_sema_rangechecks.inc -gen-arm-sme-sema-rangechecks
  SOURCE arm_sme.td
  TARGET ClangARMSmeSemaRangeChecks)
clang_tablegen(arm_sme_streaming_attrs.inc -gen-arm-sme-streaming-attrs
  SOURCE arm_sme.td
  TARGET ClangARMSmeStreamingAttrs)
clang_tablegen(arm_sme_builtins_za_state.inc -gen-arm-sme-builtin-za-state
  SOURCE arm_sme.td
  TARGET ClangARMSmeBuiltinsZAState)
clang_tablegen(arm_cde_builtins.inc -gen-arm-cde-builtin-def
  SOURCE arm_cde.td
  TARGET ClangARMCdeBuiltinsDef)
clang_tablegen(arm_cde_builtin_cg.inc -gen-arm-cde-builtin-codegen
  SOURCE arm_cde.td
  TARGET ClangARMCdeBuiltinCG)
clang_tablegen(arm_cde_builtin_sema.inc -gen-arm-cde-builtin-sema
  SOURCE arm_cde.td
  TARGET ClangARMCdeBuiltinSema)
clang_tablegen(arm_cde_builtin_aliases.inc -gen-arm-cde-builtin-aliases
  SOURCE arm_cde.td
  TARGET ClangARMCdeBuiltinAliases)
clang_tablegen(riscv_vector_builtins.inc -gen-riscv-vector-builtins
  SOURCE riscv_vector.td
  TARGET ClangRISCVVectorBuiltins)
clang_tablegen(riscv_vector_builtin_cg.inc -gen-riscv-vector-builtin-codegen
  SOURCE riscv_vector.td
  TARGET ClangRISCVVectorBuiltinCG)
clang_tablegen(riscv_vector_builtin_sema.inc -gen-riscv-vector-builtin-sema
  SOURCE riscv_vector.td
  TARGET ClangRISCVVectorBuiltinSema)
clang_tablegen(riscv_sifive_vector_builtins.inc -gen-riscv-sifive-vector-builtins
  SOURCE riscv_sifive_vector.td
  TARGET ClangRISCVSiFiveVectorBuiltins)
clang_tablegen(riscv_sifive_vector_builtin_cg.inc -gen-riscv-sifive-vector-builtin-codegen
  SOURCE riscv_sifive_vector.td
  TARGET ClangRISCVSiFiveVectorBuiltinCG)
clang_tablegen(riscv_sifive_vector_builtin_sema.inc -gen-riscv-sifive-vector-builtin-sema
  SOURCE riscv_sifive_vector.td
  TARGET ClangRISCVSiFiveVectorBuiltinSema)
clang_tablegen(riscv_andes_vector_builtins.inc -gen-riscv-andes-vector-builtins
  SOURCE riscv_andes_vector.td
  TARGET ClangRISCVAndesVectorBuiltins)
clang_tablegen(riscv_andes_vector_builtin_cg.inc -gen-riscv-andes-vector-builtin-codegen
  SOURCE riscv_andes_vector.td
  TARGET ClangRISCVAndesVectorBuiltinCG)
clang_tablegen(riscv_andes_vector_builtin_sema.inc -gen-riscv-andes-vector-builtin-sema
  SOURCE riscv_andes_vector.td
  TARGET ClangRISCVAndesVectorBuiltinSema)
