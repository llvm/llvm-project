add_subdirectory(Dialect)
add_subdirectory(Transforms)

set(LLVM_OPTIONAL_SOURCES
  ConvertFromLLVMIR.cpp
  ConvertToLLVMIR.cpp
  DataLayoutImporter.cpp
  DebugTranslation.cpp
  DebugImporter.cpp
  LoopAnnotationImporter.cpp
  LoopAnnotationTranslation.cpp
  LLVMImportInterface.cpp
  ModuleTranslation.cpp
  ModuleImport.cpp
  TypeToLLVM.cpp
  TypeFromLLVM.cpp
  )


add_aiir_translation_library(AIIRTargetLLVMIRExport
  DebugTranslation.cpp
  LoopAnnotationTranslation.cpp
  ModuleTranslation.cpp
  TypeToLLVM.cpp
  Dialect/OpenMPCommon.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Target/LLVMIR

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  FrontendOpenMP
  TransformUtils
  TargetParser

  LINK_LIBS PUBLIC
  AIIRDLTIDialect
  AIIRLLVMDialect
  AIIRLLVMIRTransforms
  AIIRTranslateLib
  AIIRTransformUtils
  )

add_aiir_translation_library(AIIRToLLVMIRTranslationRegistration
  ConvertToLLVMIR.cpp

  LINK_LIBS PUBLIC
  AIIRArmNeonToLLVMIRTranslation
  AIIRArmSMEToLLVMIRTranslation
  AIIRArmSVEToLLVMIRTranslation
  AIIRBuiltinToLLVMIRTranslation
  AIIRGPUToLLVMIRTranslation
  AIIRLLVMToLLVMIRTranslation
  AIIRNVVMToLLVMIRTranslation
  AIIROpenACCToLLVMIRTranslation
  AIIROpenMPToLLVMIRTranslation
  AIIRROCDLToLLVMIRTranslation
  AIIRPtrToLLVMIRTranslation
  AIIRSPIRVToLLVMIRTranslation
  AIIRVCIXToLLVMIRTranslation
  AIIRXeVMToLLVMIRTranslation
  )

add_aiir_translation_library(AIIRTargetLLVMIRImport
  DataLayoutImporter.cpp
  DebugImporter.cpp
  LoopAnnotationImporter.cpp
  ModuleImport.cpp
  LLVMImportInterface.cpp
  TypeFromLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Target/LLVMIR

  DEPENDS
  AIIRLLVMImportDialectInterfaceIncGen

  LINK_COMPONENTS
  Core
  IRReader

  LINK_LIBS PUBLIC
  AIIRDLTIDialect
  AIIRLLVMDialect
  AIIRTranslateLib
  )

add_aiir_translation_library(AIIRFromLLVMIRTranslationRegistration
  ConvertFromLLVMIR.cpp

  LINK_LIBS PUBLIC
  AIIRLLVMIRToLLVMTranslation
  AIIRLLVMIRToNVVMTranslation
  )
