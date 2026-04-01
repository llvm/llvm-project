set(LLVM_OPTIONAL_SOURCES
  TestDialect.cpp
  TestPatterns.cpp
  TestTraits.cpp
  TestFromLLVMIRTranslation.cpp
  TestToLLVMIRTranslation.cpp
)

set(LLVM_TARGET_DEFINITIONS TestInterfaces.td)
aiir_tablegen(TestAttrInterfaces.h.inc -gen-attr-interface-decls)
aiir_tablegen(TestAttrInterfaces.cpp.inc -gen-attr-interface-defs)
aiir_tablegen(TestTypeInterfaces.h.inc -gen-type-interface-decls)
aiir_tablegen(TestTypeInterfaces.cpp.inc -gen-type-interface-defs)
aiir_tablegen(TestOpInterfaces.h.inc -gen-op-interface-decls)
aiir_tablegen(TestOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(AIIRTestInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS TestOps.td)
aiir_tablegen(TestAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=test)
aiir_tablegen(TestAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=test)
add_public_tablegen_target(AIIRTestAttrDefIncGen)

set(LLVM_TARGET_DEFINITIONS TestTypeDefs.td)
aiir_tablegen(TestTypeDefs.h.inc -gen-typedef-decls -typedefs-dialect=test)
aiir_tablegen(TestTypeDefs.cpp.inc -gen-typedef-defs -typedefs-dialect=test)
add_public_tablegen_target(AIIRTestTypeDefIncGen)

set(LLVM_TARGET_DEFINITIONS TestEnumDefs.td)
aiir_tablegen(TestOpEnums.h.inc -gen-enum-decls)
aiir_tablegen(TestOpEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(AIIRTestEnumDefIncGen)

set(LLVM_TARGET_DEFINITIONS TestOps.td)
aiir_tablegen(TestOpsDialect.h.inc -gen-dialect-decls -dialect=test)
aiir_tablegen(TestOpsDialect.cpp.inc -gen-dialect-defs -dialect=test)
aiir_tablegen(TestPatterns.inc -gen-rewriters)
add_public_tablegen_target(AIIRTestOpsIncGen)

set(LLVM_TARGET_DEFINITIONS TestOpsSyntax.td)
aiir_tablegen(TestOpsSyntax.h.inc -gen-op-decls)
aiir_tablegen(TestOpsSyntax.cpp.inc -gen-op-defs)
add_public_tablegen_target(AIIRTestOpsSyntaxIncGen)

add_sharded_ops(TestOps 20)

# Exclude tests from libAIIR.so
add_aiir_library(AIIRTestDialect
  TestAttributes.cpp
  TestDialect.cpp
  TestFormatUtils.cpp
  TestInterfaces.cpp
  TestOpDefs.cpp
  TestOps.cpp
  TestPatterns.cpp
  TestTraits.cpp
  TestTypes.cpp
  TestOpsSyntax.cpp
  TestDialectInterfaces.cpp
  ${SHARDED_SRCS}

  EXCLUDE_FROM_LIBAIIR

  DEPENDS
  AIIRTestAttrDefIncGen
  AIIRTestEnumDefIncGen
  AIIRTestInterfaceIncGen
  AIIRTestTypeDefIncGen
  AIIRTestOpsIncGen
  AIIRTestOpsSyntaxIncGen
  AIIRTestOpsShardGen
  )
aiir_target_link_libraries(AIIRTestDialect PUBLIC
  AIIRControlFlowInterfaces
  AIIRControlFlowTransforms
  AIIRDataLayoutInterfaces
  AIIRDerivedAttributeOpInterface
  AIIRDestinationStyleOpInterface
  AIIRDialect
  AIIRDLTIDialect
  AIIRFuncDialect
  AIIRFunctionInterfaces
  AIIRFuncTransforms
  AIIRIR
  AIIRInferIntRangeInterface
  AIIRInferTypeOpInterface
  AIIRLinalgDialect
  AIIRLinalgTransforms
  AIIRPtrDialect
  AIIRLLVMDialect
  AIIRNVVMDialect
  AIIRPass
  AIIRReduce
  AIIRTensorDialect
  AIIRTransformUtils
  AIIRTransforms
  AIIRValueBoundsOpInterface
  AIIRBufferizationDialect
)

add_aiir_translation_library(AIIRTestFromLLVMIRTranslation
  TestFromLLVMIRTranslation.cpp

  EXCLUDE_FROM_LIBAIIR

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMDialect
  AIIRTestDialect
  AIIRSupport
  AIIRTargetLLVMIRImport
  AIIRLLVMIRToLLVMTranslation
)

add_aiir_translation_library(AIIRTestToLLVMIRTranslation
  TestToLLVMIRTranslation.cpp

  EXCLUDE_FROM_LIBAIIR

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRLLVMDialect
  AIIRTestDialect
  AIIRSupport
  AIIRBuiltinToLLVMIRTranslation
  AIIRLLVMToLLVMIRTranslation
)
