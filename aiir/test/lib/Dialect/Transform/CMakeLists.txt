set(LLVM_TARGET_DEFINITIONS TestTransformDialectExtension.td)
aiir_tablegen(TestTransformDialectExtension.h.inc -gen-op-decls)
aiir_tablegen(TestTransformDialectExtension.cpp.inc -gen-op-defs)
aiir_tablegen(TestTransformDialectExtensionTypes.h.inc -gen-typedef-decls -typedefs-dialect=transform)
aiir_tablegen(TestTransformDialectExtensionTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=transform)
add_public_tablegen_target(AIIRTestTransformDialectExtensionIncGen)

add_aiir_library(AIIRTestTransformDialect
  TestPassStateExtensionCommunication.cpp
  TestTransformDialectExtension.cpp
  TestTransformDialectInterpreter.cpp
  TestTransformStateExtension.cpp

  EXCLUDE_FROM_LIBAIIR

  DEPENDS
  AIIRTestTransformDialectExtensionIncGen
)
aiir_target_link_libraries(AIIRTestTransformDialect PUBLIC
  AIIRIR
  AIIRPass
  AIIRPDLDialect
  AIIRTransformDebugExtension
  AIIRTransformDialect
  AIIRTransformDialectTransforms
  AIIRTransformLoopExtension
  AIIRTransformPDLExtension
)
