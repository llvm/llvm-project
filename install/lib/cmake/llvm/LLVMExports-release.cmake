#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "LLVMDemangle" for configuration "Release"
set_property(TARGET LLVMDemangle APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDemangle PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDemangle.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMDemangle.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMDemangle )
list(APPEND _cmake_import_check_files_for_LLVMDemangle "${_IMPORT_PREFIX}/lib/libLLVMDemangle.so.21.0git" )

# Import target "LLVMSupport" for configuration "Release"
set_property(TARGET LLVMSupport APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSupport PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "zstd::libzstd_shared;LLVMDemangle"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSupport.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSupport.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSupport )
list(APPEND _cmake_import_check_files_for_LLVMSupport "${_IMPORT_PREFIX}/lib/libLLVMSupport.so.21.0git" )

# Import target "LLVMTableGen" for configuration "Release"
set_property(TARGET LLVMTableGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMTableGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMTableGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMTableGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMTableGen )
list(APPEND _cmake_import_check_files_for_LLVMTableGen "${_IMPORT_PREFIX}/lib/libLLVMTableGen.so.21.0git" )

# Import target "LLVMTableGenBasic" for configuration "Release"
set_property(TARGET LLVMTableGenBasic APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMTableGenBasic PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;LLVMTableGen"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMTableGenBasic.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMTableGenBasic.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMTableGenBasic )
list(APPEND _cmake_import_check_files_for_LLVMTableGenBasic "${_IMPORT_PREFIX}/lib/libLLVMTableGenBasic.so.21.0git" )

# Import target "LLVMTableGenCommon" for configuration "Release"
set_property(TARGET LLVMTableGenCommon APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMTableGenCommon PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMTableGenBasic;LLVMSupport;LLVMTableGen"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMTableGenCommon.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMTableGenCommon.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMTableGenCommon )
list(APPEND _cmake_import_check_files_for_LLVMTableGenCommon "${_IMPORT_PREFIX}/lib/libLLVMTableGenCommon.so.21.0git" )

# Import target "llvm-tblgen" for configuration "Release"
set_property(TARGET llvm-tblgen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-tblgen PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-tblgen"
  )

list(APPEND _cmake_import_check_targets llvm-tblgen )
list(APPEND _cmake_import_check_files_for_llvm-tblgen "${_IMPORT_PREFIX}/bin/llvm-tblgen" )

# Import target "LLVMCore" for configuration "Release"
set_property(TARGET LLVMCore APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMCore PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBinaryFormat;LLVMDemangle;LLVMRemarks;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMCore.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMCore.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMCore )
list(APPEND _cmake_import_check_files_for_LLVMCore "${_IMPORT_PREFIX}/lib/libLLVMCore.so.21.0git" )

# Import target "LLVMFuzzerCLI" for configuration "Release"
set_property(TARGET LLVMFuzzerCLI APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMFuzzerCLI PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMFuzzerCLI.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMFuzzerCLI.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMFuzzerCLI )
list(APPEND _cmake_import_check_files_for_LLVMFuzzerCLI "${_IMPORT_PREFIX}/lib/libLLVMFuzzerCLI.so.21.0git" )

# Import target "LLVMFuzzMutate" for configuration "Release"
set_property(TARGET LLVMFuzzMutate APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMFuzzMutate PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMBitReader;LLVMBitWriter;LLVMCore;LLVMScalarOpts;LLVMSupport;LLVMTargetParser;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMFuzzMutate.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMFuzzMutate.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMFuzzMutate )
list(APPEND _cmake_import_check_files_for_LLVMFuzzMutate "${_IMPORT_PREFIX}/lib/libLLVMFuzzMutate.so.21.0git" )

# Import target "LLVMFileCheck" for configuration "Release"
set_property(TARGET LLVMFileCheck APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMFileCheck PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMFileCheck.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMFileCheck.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMFileCheck )
list(APPEND _cmake_import_check_files_for_LLVMFileCheck "${_IMPORT_PREFIX}/lib/libLLVMFileCheck.so.21.0git" )

# Import target "LLVMInterfaceStub" for configuration "Release"
set_property(TARGET LLVMInterfaceStub APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMInterfaceStub PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBinaryFormat;LLVMMC;LLVMObject;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMInterfaceStub.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMInterfaceStub.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMInterfaceStub )
list(APPEND _cmake_import_check_files_for_LLVMInterfaceStub "${_IMPORT_PREFIX}/lib/libLLVMInterfaceStub.so.21.0git" )

# Import target "LLVMIRPrinter" for configuration "Release"
set_property(TARGET LLVMIRPrinter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMIRPrinter PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMIRPrinter.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMIRPrinter.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMIRPrinter )
list(APPEND _cmake_import_check_files_for_LLVMIRPrinter "${_IMPORT_PREFIX}/lib/libLLVMIRPrinter.so.21.0git" )

# Import target "LLVMIRReader" for configuration "Release"
set_property(TARGET LLVMIRReader APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMIRReader PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAsmParser;LLVMBitReader;LLVMCore;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMIRReader.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMIRReader.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMIRReader )
list(APPEND _cmake_import_check_files_for_LLVMIRReader "${_IMPORT_PREFIX}/lib/libLLVMIRReader.so.21.0git" )

# Import target "LLVMCGData" for configuration "Release"
set_property(TARGET LLVMCGData APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMCGData PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBitReader;LLVMBitWriter;LLVMCore;LLVMSupport;LLVMObject"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMCGData.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMCGData.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMCGData )
list(APPEND _cmake_import_check_files_for_LLVMCGData "${_IMPORT_PREFIX}/lib/libLLVMCGData.so.21.0git" )

# Import target "LLVMCodeGen" for configuration "Release"
set_property(TARGET LLVMCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMBitReader;LLVMBitWriter;LLVMCGData;LLVMCodeGenTypes;LLVMCore;LLVMMC;LLVMObjCARCOpts;LLVMProfileData;LLVMScalarOpts;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMCodeGen "${_IMPORT_PREFIX}/lib/libLLVMCodeGen.so.21.0git" )

# Import target "LLVMSelectionDAG" for configuration "Release"
set_property(TARGET LLVMSelectionDAG APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSelectionDAG PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMMC;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSelectionDAG.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSelectionDAG.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSelectionDAG )
list(APPEND _cmake_import_check_files_for_LLVMSelectionDAG "${_IMPORT_PREFIX}/lib/libLLVMSelectionDAG.so.21.0git" )

# Import target "LLVMAsmPrinter" for configuration "Release"
set_property(TARGET LLVMAsmPrinter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAsmPrinter PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMBinaryFormat;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMDebugInfoCodeView;LLVMDebugInfoDWARF;LLVMMC;LLVMMCParser;LLVMRemarks;LLVMSupport;LLVMTarget;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAsmPrinter.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAsmPrinter.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAsmPrinter )
list(APPEND _cmake_import_check_files_for_LLVMAsmPrinter "${_IMPORT_PREFIX}/lib/libLLVMAsmPrinter.so.21.0git" )

# Import target "LLVMMIRParser" for configuration "Release"
set_property(TARGET LLVMMIRParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMIRParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAsmParser;LLVMBinaryFormat;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMMC;LLVMSupport;LLVMTarget"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMIRParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMMIRParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMMIRParser )
list(APPEND _cmake_import_check_files_for_LLVMMIRParser "${_IMPORT_PREFIX}/lib/libLLVMMIRParser.so.21.0git" )

# Import target "LLVMGlobalISel" for configuration "Release"
set_property(TARGET LLVMGlobalISel APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMGlobalISel PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMMC;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMGlobalISel.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMGlobalISel.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMGlobalISel )
list(APPEND _cmake_import_check_files_for_LLVMGlobalISel "${_IMPORT_PREFIX}/lib/libLLVMGlobalISel.so.21.0git" )

# Import target "LLVMCodeGenTypes" for configuration "Release"
set_property(TARGET LLVMCodeGenTypes APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMCodeGenTypes PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMCodeGenTypes.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMCodeGenTypes.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMCodeGenTypes )
list(APPEND _cmake_import_check_files_for_LLVMCodeGenTypes "${_IMPORT_PREFIX}/lib/libLLVMCodeGenTypes.so.21.0git" )

# Import target "LLVMBinaryFormat" for configuration "Release"
set_property(TARGET LLVMBinaryFormat APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMBinaryFormat PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMBinaryFormat.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMBinaryFormat.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMBinaryFormat )
list(APPEND _cmake_import_check_files_for_LLVMBinaryFormat "${_IMPORT_PREFIX}/lib/libLLVMBinaryFormat.so.21.0git" )

# Import target "LLVMBitReader" for configuration "Release"
set_property(TARGET LLVMBitReader APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMBitReader PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBitstreamReader;LLVMCore;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMBitReader.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMBitReader.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMBitReader )
list(APPEND _cmake_import_check_files_for_LLVMBitReader "${_IMPORT_PREFIX}/lib/libLLVMBitReader.so.21.0git" )

# Import target "LLVMBitWriter" for configuration "Release"
set_property(TARGET LLVMBitWriter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMBitWriter PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMMC;LLVMObject;LLVMProfileData;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMBitWriter.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMBitWriter.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMBitWriter )
list(APPEND _cmake_import_check_files_for_LLVMBitWriter "${_IMPORT_PREFIX}/lib/libLLVMBitWriter.so.21.0git" )

# Import target "LLVMBitstreamReader" for configuration "Release"
set_property(TARGET LLVMBitstreamReader APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMBitstreamReader PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMBitstreamReader.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMBitstreamReader.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMBitstreamReader )
list(APPEND _cmake_import_check_files_for_LLVMBitstreamReader "${_IMPORT_PREFIX}/lib/libLLVMBitstreamReader.so.21.0git" )

# Import target "LLVMDWARFLinker" for configuration "Release"
set_property(TARGET LLVMDWARFLinker APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDWARFLinker PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBinaryFormat;LLVMCodeGen;LLVMDebugInfoDWARF;LLVMObject;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDWARFLinker.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMDWARFLinker.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMDWARFLinker )
list(APPEND _cmake_import_check_files_for_LLVMDWARFLinker "${_IMPORT_PREFIX}/lib/libLLVMDWARFLinker.so.21.0git" )

# Import target "LLVMDWARFLinkerClassic" for configuration "Release"
set_property(TARGET LLVMDWARFLinkerClassic APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDWARFLinkerClassic PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAsmPrinter;LLVMBinaryFormat;LLVMCodeGen;LLVMCodeGenTypes;LLVMDebugInfoDWARF;LLVMDWARFLinker;LLVMMC;LLVMObject;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDWARFLinkerClassic.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMDWARFLinkerClassic.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMDWARFLinkerClassic )
list(APPEND _cmake_import_check_files_for_LLVMDWARFLinkerClassic "${_IMPORT_PREFIX}/lib/libLLVMDWARFLinkerClassic.so.21.0git" )

# Import target "LLVMDWARFLinkerParallel" for configuration "Release"
set_property(TARGET LLVMDWARFLinkerParallel APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDWARFLinkerParallel PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAsmPrinter;LLVMBinaryFormat;LLVMCodeGen;LLVMDebugInfoDWARF;LLVMDWARFLinker;LLVMMC;LLVMObject;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDWARFLinkerParallel.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMDWARFLinkerParallel.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMDWARFLinkerParallel )
list(APPEND _cmake_import_check_files_for_LLVMDWARFLinkerParallel "${_IMPORT_PREFIX}/lib/libLLVMDWARFLinkerParallel.so.21.0git" )

# Import target "LLVMExtensions" for configuration "Release"
set_property(TARGET LLVMExtensions APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMExtensions PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMExtensions.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMExtensions.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMExtensions )
list(APPEND _cmake_import_check_files_for_LLVMExtensions "${_IMPORT_PREFIX}/lib/libLLVMExtensions.so.21.0git" )

# Import target "LLVMFrontendAtomic" for configuration "Release"
set_property(TARGET LLVMFrontendAtomic APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMFrontendAtomic PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMSupport;LLVMAnalysis"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMFrontendAtomic.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMFrontendAtomic.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMFrontendAtomic )
list(APPEND _cmake_import_check_files_for_LLVMFrontendAtomic "${_IMPORT_PREFIX}/lib/libLLVMFrontendAtomic.so.21.0git" )

# Import target "LLVMFrontendDriver" for configuration "Release"
set_property(TARGET LLVMFrontendDriver APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMFrontendDriver PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMSupport;LLVMAnalysis"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMFrontendDriver.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMFrontendDriver.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMFrontendDriver )
list(APPEND _cmake_import_check_files_for_LLVMFrontendDriver "${_IMPORT_PREFIX}/lib/libLLVMFrontendDriver.so.21.0git" )

# Import target "LLVMFrontendHLSL" for configuration "Release"
set_property(TARGET LLVMFrontendHLSL APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMFrontendHLSL PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMFrontendHLSL.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMFrontendHLSL.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMFrontendHLSL )
list(APPEND _cmake_import_check_files_for_LLVMFrontendHLSL "${_IMPORT_PREFIX}/lib/libLLVMFrontendHLSL.so.21.0git" )

# Import target "LLVMFrontendOpenACC" for configuration "Release"
set_property(TARGET LLVMFrontendOpenACC APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMFrontendOpenACC PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMFrontendOpenACC.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMFrontendOpenACC.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMFrontendOpenACC )
list(APPEND _cmake_import_check_files_for_LLVMFrontendOpenACC "${_IMPORT_PREFIX}/lib/libLLVMFrontendOpenACC.so.21.0git" )

# Import target "LLVMFrontendOpenMP" for configuration "Release"
set_property(TARGET LLVMFrontendOpenMP APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMFrontendOpenMP PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMSupport;LLVMTargetParser;LLVMTransformUtils;LLVMAnalysis;LLVMDemangle;LLVMMC;LLVMScalarOpts;LLVMBitReader;LLVMFrontendOffloading;LLVMFrontendAtomic"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMFrontendOpenMP.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMFrontendOpenMP.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMFrontendOpenMP )
list(APPEND _cmake_import_check_files_for_LLVMFrontendOpenMP "${_IMPORT_PREFIX}/lib/libLLVMFrontendOpenMP.so.21.0git" )

# Import target "LLVMFrontendOffloading" for configuration "Release"
set_property(TARGET LLVMFrontendOffloading APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMFrontendOffloading PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMBinaryFormat;LLVMObject;LLVMObjectYAML;LLVMSupport;LLVMTransformUtils;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMFrontendOffloading.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMFrontendOffloading.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMFrontendOffloading )
list(APPEND _cmake_import_check_files_for_LLVMFrontendOffloading "${_IMPORT_PREFIX}/lib/libLLVMFrontendOffloading.so.21.0git" )

# Import target "LLVMTransformUtils" for configuration "Release"
set_property(TARGET LLVMTransformUtils APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMTransformUtils PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMTransformUtils.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMTransformUtils.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMTransformUtils )
list(APPEND _cmake_import_check_files_for_LLVMTransformUtils "${_IMPORT_PREFIX}/lib/libLLVMTransformUtils.so.21.0git" )

# Import target "LLVMInstrumentation" for configuration "Release"
set_property(TARGET LLVMInstrumentation APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMInstrumentation PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMDemangle;LLVMMC;LLVMSupport;LLVMTargetParser;LLVMTransformUtils;LLVMProfileData"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMInstrumentation.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMInstrumentation.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMInstrumentation )
list(APPEND _cmake_import_check_files_for_LLVMInstrumentation "${_IMPORT_PREFIX}/lib/libLLVMInstrumentation.so.21.0git" )

# Import target "LLVMAggressiveInstCombine" for configuration "Release"
set_property(TARGET LLVMAggressiveInstCombine APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAggressiveInstCombine PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMSupport;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAggressiveInstCombine.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAggressiveInstCombine.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAggressiveInstCombine )
list(APPEND _cmake_import_check_files_for_LLVMAggressiveInstCombine "${_IMPORT_PREFIX}/lib/libLLVMAggressiveInstCombine.so.21.0git" )

# Import target "LLVMInstCombine" for configuration "Release"
set_property(TARGET LLVMInstCombine APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMInstCombine PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMSupport;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMInstCombine.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMInstCombine.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMInstCombine )
list(APPEND _cmake_import_check_files_for_LLVMInstCombine "${_IMPORT_PREFIX}/lib/libLLVMInstCombine.so.21.0git" )

# Import target "LLVMScalarOpts" for configuration "Release"
set_property(TARGET LLVMScalarOpts APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMScalarOpts PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAggressiveInstCombine;LLVMAnalysis;LLVMCore;LLVMInstCombine;LLVMSupport;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMScalarOpts.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMScalarOpts.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMScalarOpts )
list(APPEND _cmake_import_check_files_for_LLVMScalarOpts "${_IMPORT_PREFIX}/lib/libLLVMScalarOpts.so.21.0git" )

# Import target "LLVMipo" for configuration "Release"
set_property(TARGET LLVMipo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMipo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAggressiveInstCombine;LLVMAnalysis;LLVMBitReader;LLVMBitWriter;LLVMCore;LLVMFrontendOpenMP;LLVMInstCombine;LLVMIRReader;LLVMDemangle;LLVMLinker;LLVMObject;LLVMProfileData;LLVMScalarOpts;LLVMSupport;LLVMTargetParser;LLVMTransformUtils;LLVMVectorize;LLVMInstrumentation"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMipo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMipo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMipo )
list(APPEND _cmake_import_check_files_for_LLVMipo "${_IMPORT_PREFIX}/lib/libLLVMipo.so.21.0git" )

# Import target "LLVMVectorize" for configuration "Release"
set_property(TARGET LLVMVectorize APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMVectorize PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMSupport;LLVMTransformUtils;LLVMSandboxIR"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMVectorize.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMVectorize.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMVectorize )
list(APPEND _cmake_import_check_files_for_LLVMVectorize "${_IMPORT_PREFIX}/lib/libLLVMVectorize.so.21.0git" )

# Import target "LLVMObjCARCOpts" for configuration "Release"
set_property(TARGET LLVMObjCARCOpts APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMObjCARCOpts PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMSupport;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMObjCARCOpts.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMObjCARCOpts.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMObjCARCOpts )
list(APPEND _cmake_import_check_files_for_LLVMObjCARCOpts "${_IMPORT_PREFIX}/lib/libLLVMObjCARCOpts.so.21.0git" )

# Import target "LLVMCoroutines" for configuration "Release"
set_property(TARGET LLVMCoroutines APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMCoroutines PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMipo;LLVMScalarOpts;LLVMSupport;LLVMTransformUtils;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMCoroutines.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMCoroutines.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMCoroutines )
list(APPEND _cmake_import_check_files_for_LLVMCoroutines "${_IMPORT_PREFIX}/lib/libLLVMCoroutines.so.21.0git" )

# Import target "LLVMCFGuard" for configuration "Release"
set_property(TARGET LLVMCFGuard APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMCFGuard PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMCFGuard.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMCFGuard.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMCFGuard )
list(APPEND _cmake_import_check_files_for_LLVMCFGuard "${_IMPORT_PREFIX}/lib/libLLVMCFGuard.so.21.0git" )

# Import target "LLVMHipStdPar" for configuration "Release"
set_property(TARGET LLVMHipStdPar APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMHipStdPar PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMSupport;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMHipStdPar.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMHipStdPar.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMHipStdPar )
list(APPEND _cmake_import_check_files_for_LLVMHipStdPar "${_IMPORT_PREFIX}/lib/libLLVMHipStdPar.so.21.0git" )

# Import target "LLVMLinker" for configuration "Release"
set_property(TARGET LLVMLinker APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLinker PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMObject;LLVMSupport;LLVMTransformUtils;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLinker.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMLinker.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMLinker )
list(APPEND _cmake_import_check_files_for_LLVMLinker "${_IMPORT_PREFIX}/lib/libLLVMLinker.so.21.0git" )

# Import target "LLVMAnalysis" for configuration "Release"
set_property(TARGET LLVMAnalysis APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAnalysis PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBinaryFormat;LLVMCore;LLVMObject;LLVMProfileData;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAnalysis.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAnalysis.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAnalysis )
list(APPEND _cmake_import_check_files_for_LLVMAnalysis "${_IMPORT_PREFIX}/lib/libLLVMAnalysis.so.21.0git" )

# Import target "LLVMLTO" for configuration "Release"
set_property(TARGET LLVMLTO APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLTO PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAggressiveInstCombine;LLVMAnalysis;LLVMBinaryFormat;LLVMBitReader;LLVMBitWriter;LLVMCGData;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMExtensions;LLVMipo;LLVMInstCombine;LLVMInstrumentation;LLVMLinker;LLVMMC;LLVMObjCARCOpts;LLVMObject;LLVMPasses;LLVMRemarks;LLVMScalarOpts;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLTO.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMLTO.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMLTO )
list(APPEND _cmake_import_check_files_for_LLVMLTO "${_IMPORT_PREFIX}/lib/libLLVMLTO.so.21.0git" )

# Import target "LLVMMC" for configuration "Release"
set_property(TARGET LLVMMC APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMC PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;LLVMTargetParser;LLVMBinaryFormat"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMC.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMMC.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMMC )
list(APPEND _cmake_import_check_files_for_LLVMMC "${_IMPORT_PREFIX}/lib/libLLVMMC.so.21.0git" )

# Import target "LLVMMCParser" for configuration "Release"
set_property(TARGET LLVMMCParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMCParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMCParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMMCParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMMCParser )
list(APPEND _cmake_import_check_files_for_LLVMMCParser "${_IMPORT_PREFIX}/lib/libLLVMMCParser.so.21.0git" )

# Import target "LLVMMCDisassembler" for configuration "Release"
set_property(TARGET LLVMMCDisassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMCDisassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMCDisassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMMCDisassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMMCDisassembler )
list(APPEND _cmake_import_check_files_for_LLVMMCDisassembler "${_IMPORT_PREFIX}/lib/libLLVMMCDisassembler.so.21.0git" )

# Import target "LLVMMCA" for configuration "Release"
set_property(TARGET LLVMMCA APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMCA PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMCA.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMMCA.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMMCA )
list(APPEND _cmake_import_check_files_for_LLVMMCA "${_IMPORT_PREFIX}/lib/libLLVMMCA.so.21.0git" )

# Import target "LLVMObjCopy" for configuration "Release"
set_property(TARGET LLVMObjCopy APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMObjCopy PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBinaryFormat;LLVMObject;LLVMSupport;LLVMMC"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMObjCopy.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMObjCopy.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMObjCopy )
list(APPEND _cmake_import_check_files_for_LLVMObjCopy "${_IMPORT_PREFIX}/lib/libLLVMObjCopy.so.21.0git" )

# Import target "LLVMObject" for configuration "Release"
set_property(TARGET LLVMObject APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMObject PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBitReader;LLVMCore;LLVMMC;LLVMIRReader;LLVMBinaryFormat;LLVMMCParser;LLVMSupport;LLVMTargetParser;LLVMTextAPI"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMObject.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMObject.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMObject )
list(APPEND _cmake_import_check_files_for_LLVMObject "${_IMPORT_PREFIX}/lib/libLLVMObject.so.21.0git" )

# Import target "LLVMObjectYAML" for configuration "Release"
set_property(TARGET LLVMObjectYAML APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMObjectYAML PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBinaryFormat;LLVMObject;LLVMSupport;LLVMTargetParser;LLVMDebugInfoCodeView;LLVMMC"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMObjectYAML.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMObjectYAML.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMObjectYAML )
list(APPEND _cmake_import_check_files_for_LLVMObjectYAML "${_IMPORT_PREFIX}/lib/libLLVMObjectYAML.so.21.0git" )

# Import target "LLVMOption" for configuration "Release"
set_property(TARGET LLVMOption APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMOption PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMOption.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMOption.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMOption )
list(APPEND _cmake_import_check_files_for_LLVMOption "${_IMPORT_PREFIX}/lib/libLLVMOption.so.21.0git" )

# Import target "LLVMRemarks" for configuration "Release"
set_property(TARGET LLVMRemarks APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMRemarks PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBitstreamReader;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMRemarks.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMRemarks.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMRemarks )
list(APPEND _cmake_import_check_files_for_LLVMRemarks "${_IMPORT_PREFIX}/lib/libLLVMRemarks.so.21.0git" )

# Import target "LLVMDebuginfod" for configuration "Release"
set_property(TARGET LLVMDebuginfod APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDebuginfod PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;LLVMSymbolize;LLVMDebugInfoDWARF;LLVMBinaryFormat;LLVMObject"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDebuginfod.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMDebuginfod.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMDebuginfod )
list(APPEND _cmake_import_check_files_for_LLVMDebuginfod "${_IMPORT_PREFIX}/lib/libLLVMDebuginfod.so.21.0git" )

# Import target "LLVMDebugInfoDWARF" for configuration "Release"
set_property(TARGET LLVMDebugInfoDWARF APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDebugInfoDWARF PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBinaryFormat;LLVMObject;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoDWARF.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMDebugInfoDWARF.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMDebugInfoDWARF )
list(APPEND _cmake_import_check_files_for_LLVMDebugInfoDWARF "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoDWARF.so.21.0git" )

# Import target "LLVMDebugInfoGSYM" for configuration "Release"
set_property(TARGET LLVMDebugInfoGSYM APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDebugInfoGSYM PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMObject;LLVMSupport;LLVMTargetParser;LLVMDebugInfoDWARF"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoGSYM.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMDebugInfoGSYM.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMDebugInfoGSYM )
list(APPEND _cmake_import_check_files_for_LLVMDebugInfoGSYM "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoGSYM.so.21.0git" )

# Import target "LLVMDebugInfoLogicalView" for configuration "Release"
set_property(TARGET LLVMDebugInfoLogicalView APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDebugInfoLogicalView PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBinaryFormat;LLVMDemangle;LLVMObject;LLVMMC;LLVMSupport;LLVMTargetParser;LLVMDebugInfoDWARF;LLVMDebugInfoCodeView;LLVMDebugInfoPDB"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoLogicalView.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMDebugInfoLogicalView.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMDebugInfoLogicalView )
list(APPEND _cmake_import_check_files_for_LLVMDebugInfoLogicalView "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoLogicalView.so.21.0git" )

# Import target "LLVMDebugInfoMSF" for configuration "Release"
set_property(TARGET LLVMDebugInfoMSF APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDebugInfoMSF PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoMSF.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMDebugInfoMSF.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMDebugInfoMSF )
list(APPEND _cmake_import_check_files_for_LLVMDebugInfoMSF "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoMSF.so.21.0git" )

# Import target "LLVMDebugInfoCodeView" for configuration "Release"
set_property(TARGET LLVMDebugInfoCodeView APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDebugInfoCodeView PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoCodeView.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMDebugInfoCodeView.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMDebugInfoCodeView )
list(APPEND _cmake_import_check_files_for_LLVMDebugInfoCodeView "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoCodeView.so.21.0git" )

# Import target "LLVMDebugInfoPDB" for configuration "Release"
set_property(TARGET LLVMDebugInfoPDB APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDebugInfoPDB PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBinaryFormat;LLVMObject;LLVMSupport;LLVMDebugInfoCodeView;LLVMDebugInfoMSF"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoPDB.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMDebugInfoPDB.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMDebugInfoPDB )
list(APPEND _cmake_import_check_files_for_LLVMDebugInfoPDB "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoPDB.so.21.0git" )

# Import target "LLVMSymbolize" for configuration "Release"
set_property(TARGET LLVMSymbolize APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSymbolize PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMDebugInfoDWARF;LLVMDebugInfoPDB;LLVMDebugInfoBTF;LLVMObject;LLVMSupport;LLVMDemangle;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSymbolize.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSymbolize.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSymbolize )
list(APPEND _cmake_import_check_files_for_LLVMSymbolize "${_IMPORT_PREFIX}/lib/libLLVMSymbolize.so.21.0git" )

# Import target "LLVMDebugInfoBTF" for configuration "Release"
set_property(TARGET LLVMDebugInfoBTF APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDebugInfoBTF PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoBTF.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMDebugInfoBTF.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMDebugInfoBTF )
list(APPEND _cmake_import_check_files_for_LLVMDebugInfoBTF "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoBTF.so.21.0git" )

# Import target "LLVMDWP" for configuration "Release"
set_property(TARGET LLVMDWP APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDWP PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMDebugInfoDWARF;LLVMMC;LLVMObject;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDWP.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMDWP.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMDWP )
list(APPEND _cmake_import_check_files_for_LLVMDWP "${_IMPORT_PREFIX}/lib/libLLVMDWP.so.21.0git" )

# Import target "LLVMExecutionEngine" for configuration "Release"
set_property(TARGET LLVMExecutionEngine APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMExecutionEngine PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMMC;LLVMObject;LLVMOrcTargetProcess;LLVMSupport;LLVMTarget;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMExecutionEngine.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMExecutionEngine.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMExecutionEngine )
list(APPEND _cmake_import_check_files_for_LLVMExecutionEngine "${_IMPORT_PREFIX}/lib/libLLVMExecutionEngine.so.21.0git" )

# Import target "LLVMInterpreter" for configuration "Release"
set_property(TARGET LLVMInterpreter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMInterpreter PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGen;LLVMCore;LLVMExecutionEngine;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMInterpreter.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMInterpreter.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMInterpreter )
list(APPEND _cmake_import_check_files_for_LLVMInterpreter "${_IMPORT_PREFIX}/lib/libLLVMInterpreter.so.21.0git" )

# Import target "LLVMJITLink" for configuration "Release"
set_property(TARGET LLVMJITLink APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMJITLink PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMObject;LLVMOrcShared;LLVMOrcTargetProcess;LLVMSupport;LLVMTargetParser;LLVMBinaryFormat;LLVMOption"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMJITLink.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMJITLink.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMJITLink )
list(APPEND _cmake_import_check_files_for_LLVMJITLink "${_IMPORT_PREFIX}/lib/libLLVMJITLink.so.21.0git" )

# Import target "LLVMMCJIT" for configuration "Release"
set_property(TARGET LLVMMCJIT APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMCJIT PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMExecutionEngine;LLVMObject;LLVMRuntimeDyld;LLVMSupport;LLVMTarget"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMCJIT.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMMCJIT.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMMCJIT )
list(APPEND _cmake_import_check_files_for_LLVMMCJIT "${_IMPORT_PREFIX}/lib/libLLVMMCJIT.so.21.0git" )

# Import target "LLVMOrcJIT" for configuration "Release"
set_property(TARGET LLVMOrcJIT APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMOrcJIT PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMBitReader;LLVMBitWriter;LLVMPasses;LLVMBinaryFormat;LLVMCore;LLVMExecutionEngine;LLVMJITLink;LLVMObject;LLVMOrcShared;LLVMOrcTargetProcess;LLVMWindowsDriver;LLVMMC;LLVMRuntimeDyld;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTextAPI;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMOrcJIT.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMOrcJIT.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMOrcJIT )
list(APPEND _cmake_import_check_files_for_LLVMOrcJIT "${_IMPORT_PREFIX}/lib/libLLVMOrcJIT.so.21.0git" )

# Import target "LLVMOrcDebugging" for configuration "Release"
set_property(TARGET LLVMOrcDebugging APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMOrcDebugging PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBinaryFormat;LLVMDebugInfoDWARF;LLVMJITLink;LLVMOrcJIT;LLVMOrcShared;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMOrcDebugging.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMOrcDebugging.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMOrcDebugging )
list(APPEND _cmake_import_check_files_for_LLVMOrcDebugging "${_IMPORT_PREFIX}/lib/libLLVMOrcDebugging.so.21.0git" )

# Import target "LLVMOrcShared" for configuration "Release"
set_property(TARGET LLVMOrcShared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMOrcShared PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMOrcShared.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMOrcShared.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMOrcShared )
list(APPEND _cmake_import_check_files_for_LLVMOrcShared "${_IMPORT_PREFIX}/lib/libLLVMOrcShared.so.21.0git" )

# Import target "LLVMOrcTargetProcess" for configuration "Release"
set_property(TARGET LLVMOrcTargetProcess APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMOrcTargetProcess PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMOrcShared;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMOrcTargetProcess.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMOrcTargetProcess.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMOrcTargetProcess )
list(APPEND _cmake_import_check_files_for_LLVMOrcTargetProcess "${_IMPORT_PREFIX}/lib/libLLVMOrcTargetProcess.so.21.0git" )

# Import target "LLVMRuntimeDyld" for configuration "Release"
set_property(TARGET LLVMRuntimeDyld APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMRuntimeDyld PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMMC;LLVMObject;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMRuntimeDyld.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMRuntimeDyld.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMRuntimeDyld )
list(APPEND _cmake_import_check_files_for_LLVMRuntimeDyld "${_IMPORT_PREFIX}/lib/libLLVMRuntimeDyld.so.21.0git" )

# Import target "LLVMTarget" for configuration "Release"
set_property(TARGET LLVMTarget APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMTarget PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMMC;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMTarget.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMTarget.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMTarget )
list(APPEND _cmake_import_check_files_for_LLVMTarget "${_IMPORT_PREFIX}/lib/libLLVMTarget.so.21.0git" )

# Import target "LLVMAArch64CodeGen" for configuration "Release"
set_property(TARGET LLVMAArch64CodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAArch64CodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAArch64Desc;LLVMAArch64Info;LLVMAArch64Utils;LLVMAnalysis;LLVMAsmPrinter;LLVMCFGuard;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMGlobalISel;LLVMMC;LLVMScalarOpts;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils;LLVMVectorize"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAArch64CodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAArch64CodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAArch64CodeGen )
list(APPEND _cmake_import_check_files_for_LLVMAArch64CodeGen "${_IMPORT_PREFIX}/lib/libLLVMAArch64CodeGen.so.21.0git" )

# Import target "LLVMAArch64AsmParser" for configuration "Release"
set_property(TARGET LLVMAArch64AsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAArch64AsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAArch64Desc;LLVMAArch64Info;LLVMAArch64Utils;LLVMCodeGenTypes;LLVMMC;LLVMMCParser;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAArch64AsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAArch64AsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAArch64AsmParser )
list(APPEND _cmake_import_check_files_for_LLVMAArch64AsmParser "${_IMPORT_PREFIX}/lib/libLLVMAArch64AsmParser.so.21.0git" )

# Import target "LLVMAArch64Disassembler" for configuration "Release"
set_property(TARGET LLVMAArch64Disassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAArch64Disassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAArch64Desc;LLVMAArch64Info;LLVMAArch64Utils;LLVMMC;LLVMMCDisassembler;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAArch64Disassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAArch64Disassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAArch64Disassembler )
list(APPEND _cmake_import_check_files_for_LLVMAArch64Disassembler "${_IMPORT_PREFIX}/lib/libLLVMAArch64Disassembler.so.21.0git" )

# Import target "LLVMAArch64Desc" for configuration "Release"
set_property(TARGET LLVMAArch64Desc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAArch64Desc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAArch64Info;LLVMAArch64Utils;LLVMBinaryFormat;LLVMCodeGenTypes;LLVMMC;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAArch64Desc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAArch64Desc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAArch64Desc )
list(APPEND _cmake_import_check_files_for_LLVMAArch64Desc "${_IMPORT_PREFIX}/lib/libLLVMAArch64Desc.so.21.0git" )

# Import target "LLVMAArch64Info" for configuration "Release"
set_property(TARGET LLVMAArch64Info APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAArch64Info PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAArch64Info.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAArch64Info.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAArch64Info )
list(APPEND _cmake_import_check_files_for_LLVMAArch64Info "${_IMPORT_PREFIX}/lib/libLLVMAArch64Info.so.21.0git" )

# Import target "LLVMAArch64Utils" for configuration "Release"
set_property(TARGET LLVMAArch64Utils APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAArch64Utils PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;LLVMCore"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAArch64Utils.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAArch64Utils.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAArch64Utils )
list(APPEND _cmake_import_check_files_for_LLVMAArch64Utils "${_IMPORT_PREFIX}/lib/libLLVMAArch64Utils.so.21.0git" )

# Import target "LLVMAMDGPUCodeGen" for configuration "Release"
set_property(TARGET LLVMAMDGPUCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAMDGPUCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAMDGPUDesc;LLVMAMDGPUInfo;LLVMAMDGPUUtils;LLVMAnalysis;LLVMAsmPrinter;LLVMBinaryFormat;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMGlobalISel;LLVMHipStdPar;LLVMipo;LLVMIRPrinter;LLVMInstrumentation;LLVMMC;LLVMMIRParser;LLVMPasses;LLVMScalarOpts;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils;LLVMVectorize"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAMDGPUCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAMDGPUCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAMDGPUCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMAMDGPUCodeGen "${_IMPORT_PREFIX}/lib/libLLVMAMDGPUCodeGen.so.21.0git" )

# Import target "LLVMAMDGPUAsmParser" for configuration "Release"
set_property(TARGET LLVMAMDGPUAsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAMDGPUAsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAMDGPUDesc;LLVMAMDGPUInfo;LLVMAMDGPUUtils;LLVMCodeGenTypes;LLVMMC;LLVMMCParser;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAMDGPUAsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAMDGPUAsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAMDGPUAsmParser )
list(APPEND _cmake_import_check_files_for_LLVMAMDGPUAsmParser "${_IMPORT_PREFIX}/lib/libLLVMAMDGPUAsmParser.so.21.0git" )

# Import target "LLVMAMDGPUDisassembler" for configuration "Release"
set_property(TARGET LLVMAMDGPUDisassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAMDGPUDisassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAMDGPUDesc;LLVMAMDGPUInfo;LLVMAMDGPUUtils;LLVMCodeGenTypes;LLVMMC;LLVMMCDisassembler;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAMDGPUDisassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAMDGPUDisassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAMDGPUDisassembler )
list(APPEND _cmake_import_check_files_for_LLVMAMDGPUDisassembler "${_IMPORT_PREFIX}/lib/libLLVMAMDGPUDisassembler.so.21.0git" )

# Import target "LLVMAMDGPUTargetMCA" for configuration "Release"
set_property(TARGET LLVMAMDGPUTargetMCA APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAMDGPUTargetMCA PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAMDGPUDesc;LLVMAMDGPUInfo;LLVMAMDGPUUtils;LLVMCodeGenTypes;LLVMMC;LLVMMCA;LLVMMCParser;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAMDGPUTargetMCA.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAMDGPUTargetMCA.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAMDGPUTargetMCA )
list(APPEND _cmake_import_check_files_for_LLVMAMDGPUTargetMCA "${_IMPORT_PREFIX}/lib/libLLVMAMDGPUTargetMCA.so.21.0git" )

# Import target "LLVMAMDGPUDesc" for configuration "Release"
set_property(TARGET LLVMAMDGPUDesc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAMDGPUDesc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAMDGPUInfo;LLVMAMDGPUUtils;LLVMBinaryFormat;LLVMCodeGenTypes;LLVMCore;LLVMMC;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAMDGPUDesc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAMDGPUDesc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAMDGPUDesc )
list(APPEND _cmake_import_check_files_for_LLVMAMDGPUDesc "${_IMPORT_PREFIX}/lib/libLLVMAMDGPUDesc.so.21.0git" )

# Import target "LLVMAMDGPUInfo" for configuration "Release"
set_property(TARGET LLVMAMDGPUInfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAMDGPUInfo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAMDGPUInfo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAMDGPUInfo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAMDGPUInfo )
list(APPEND _cmake_import_check_files_for_LLVMAMDGPUInfo "${_IMPORT_PREFIX}/lib/libLLVMAMDGPUInfo.so.21.0git" )

# Import target "LLVMAMDGPUUtils" for configuration "Release"
set_property(TARGET LLVMAMDGPUUtils APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAMDGPUUtils PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMBinaryFormat;LLVMCodeGenTypes;LLVMCore;LLVMMC;LLVMMCParser;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAMDGPUUtils.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAMDGPUUtils.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAMDGPUUtils )
list(APPEND _cmake_import_check_files_for_LLVMAMDGPUUtils "${_IMPORT_PREFIX}/lib/libLLVMAMDGPUUtils.so.21.0git" )

# Import target "LLVMARMCodeGen" for configuration "Release"
set_property(TARGET LLVMARMCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMARMCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMARMDesc;LLVMARMInfo;LLVMARMUtils;LLVMAnalysis;LLVMAsmPrinter;LLVMCFGuard;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMGlobalISel;LLVMipo;LLVMMC;LLVMScalarOpts;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMARMCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMARMCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMARMCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMARMCodeGen "${_IMPORT_PREFIX}/lib/libLLVMARMCodeGen.so.21.0git" )

# Import target "LLVMARMAsmParser" for configuration "Release"
set_property(TARGET LLVMARMAsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMARMAsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMARMDesc;LLVMARMInfo;LLVMARMUtils;LLVMCodeGenTypes;LLVMMC;LLVMMCParser;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMARMAsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMARMAsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMARMAsmParser )
list(APPEND _cmake_import_check_files_for_LLVMARMAsmParser "${_IMPORT_PREFIX}/lib/libLLVMARMAsmParser.so.21.0git" )

# Import target "LLVMARMDisassembler" for configuration "Release"
set_property(TARGET LLVMARMDisassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMARMDisassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMARMDesc;LLVMARMInfo;LLVMARMUtils;LLVMCodeGenTypes;LLVMMC;LLVMMCDisassembler;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMARMDisassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMARMDisassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMARMDisassembler )
list(APPEND _cmake_import_check_files_for_LLVMARMDisassembler "${_IMPORT_PREFIX}/lib/libLLVMARMDisassembler.so.21.0git" )

# Import target "LLVMARMDesc" for configuration "Release"
set_property(TARGET LLVMARMDesc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMARMDesc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMARMInfo;LLVMARMUtils;LLVMBinaryFormat;LLVMCodeGenTypes;LLVMMC;LLVMMCDisassembler;LLVMObject;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMARMDesc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMARMDesc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMARMDesc )
list(APPEND _cmake_import_check_files_for_LLVMARMDesc "${_IMPORT_PREFIX}/lib/libLLVMARMDesc.so.21.0git" )

# Import target "LLVMARMInfo" for configuration "Release"
set_property(TARGET LLVMARMInfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMARMInfo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMARMInfo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMARMInfo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMARMInfo )
list(APPEND _cmake_import_check_files_for_LLVMARMInfo "${_IMPORT_PREFIX}/lib/libLLVMARMInfo.so.21.0git" )

# Import target "LLVMARMUtils" for configuration "Release"
set_property(TARGET LLVMARMUtils APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMARMUtils PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMARMUtils.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMARMUtils.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMARMUtils )
list(APPEND _cmake_import_check_files_for_LLVMARMUtils "${_IMPORT_PREFIX}/lib/libLLVMARMUtils.so.21.0git" )

# Import target "LLVMAVRCodeGen" for configuration "Release"
set_property(TARGET LLVMAVRCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAVRCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAVRDesc;LLVMAVRInfo;LLVMAsmPrinter;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMMC;LLVMSelectionDAG;LLVMSupport;LLVMTarget"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAVRCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAVRCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAVRCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMAVRCodeGen "${_IMPORT_PREFIX}/lib/libLLVMAVRCodeGen.so.21.0git" )

# Import target "LLVMAVRAsmParser" for configuration "Release"
set_property(TARGET LLVMAVRAsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAVRAsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAVRDesc;LLVMAVRInfo;LLVMCodeGenTypes;LLVMMC;LLVMMCParser;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAVRAsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAVRAsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAVRAsmParser )
list(APPEND _cmake_import_check_files_for_LLVMAVRAsmParser "${_IMPORT_PREFIX}/lib/libLLVMAVRAsmParser.so.21.0git" )

# Import target "LLVMAVRDisassembler" for configuration "Release"
set_property(TARGET LLVMAVRDisassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAVRDisassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAVRInfo;LLVMCodeGenTypes;LLVMMC;LLVMMCDisassembler;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAVRDisassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAVRDisassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAVRDisassembler )
list(APPEND _cmake_import_check_files_for_LLVMAVRDisassembler "${_IMPORT_PREFIX}/lib/libLLVMAVRDisassembler.so.21.0git" )

# Import target "LLVMAVRDesc" for configuration "Release"
set_property(TARGET LLVMAVRDesc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAVRDesc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMAVRInfo;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAVRDesc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAVRDesc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAVRDesc )
list(APPEND _cmake_import_check_files_for_LLVMAVRDesc "${_IMPORT_PREFIX}/lib/libLLVMAVRDesc.so.21.0git" )

# Import target "LLVMAVRInfo" for configuration "Release"
set_property(TARGET LLVMAVRInfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAVRInfo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAVRInfo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAVRInfo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAVRInfo )
list(APPEND _cmake_import_check_files_for_LLVMAVRInfo "${_IMPORT_PREFIX}/lib/libLLVMAVRInfo.so.21.0git" )

# Import target "LLVMBPFCodeGen" for configuration "Release"
set_property(TARGET LLVMBPFCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMBPFCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMAsmPrinter;LLVMBPFDesc;LLVMBPFInfo;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMGlobalISel;LLVMipo;LLVMMC;LLVMPasses;LLVMScalarOpts;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMBPFCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMBPFCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMBPFCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMBPFCodeGen "${_IMPORT_PREFIX}/lib/libLLVMBPFCodeGen.so.21.0git" )

# Import target "LLVMBPFAsmParser" for configuration "Release"
set_property(TARGET LLVMBPFAsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMBPFAsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMMCParser;LLVMBPFDesc;LLVMBPFInfo;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMBPFAsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMBPFAsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMBPFAsmParser )
list(APPEND _cmake_import_check_files_for_LLVMBPFAsmParser "${_IMPORT_PREFIX}/lib/libLLVMBPFAsmParser.so.21.0git" )

# Import target "LLVMBPFDisassembler" for configuration "Release"
set_property(TARGET LLVMBPFDisassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMBPFDisassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMCDisassembler;LLVMMC;LLVMBPFInfo;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMBPFDisassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMBPFDisassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMBPFDisassembler )
list(APPEND _cmake_import_check_files_for_LLVMBPFDisassembler "${_IMPORT_PREFIX}/lib/libLLVMBPFDisassembler.so.21.0git" )

# Import target "LLVMBPFDesc" for configuration "Release"
set_property(TARGET LLVMBPFDesc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMBPFDesc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMBPFInfo;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMBPFDesc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMBPFDesc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMBPFDesc )
list(APPEND _cmake_import_check_files_for_LLVMBPFDesc "${_IMPORT_PREFIX}/lib/libLLVMBPFDesc.so.21.0git" )

# Import target "LLVMBPFInfo" for configuration "Release"
set_property(TARGET LLVMBPFInfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMBPFInfo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMBPFInfo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMBPFInfo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMBPFInfo )
list(APPEND _cmake_import_check_files_for_LLVMBPFInfo "${_IMPORT_PREFIX}/lib/libLLVMBPFInfo.so.21.0git" )

# Import target "LLVMHexagonCodeGen" for configuration "Release"
set_property(TARGET LLVMHexagonCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMHexagonCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMAsmPrinter;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMHexagonAsmParser;LLVMHexagonDesc;LLVMHexagonInfo;LLVMipo;LLVMMC;LLVMScalarOpts;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMHexagonCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMHexagonCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMHexagonCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMHexagonCodeGen "${_IMPORT_PREFIX}/lib/libLLVMHexagonCodeGen.so.21.0git" )

# Import target "LLVMHexagonAsmParser" for configuration "Release"
set_property(TARGET LLVMHexagonAsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMHexagonAsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMMCParser;LLVMSupport;LLVMHexagonDesc;LLVMHexagonInfo"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMHexagonAsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMHexagonAsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMHexagonAsmParser )
list(APPEND _cmake_import_check_files_for_LLVMHexagonAsmParser "${_IMPORT_PREFIX}/lib/libLLVMHexagonAsmParser.so.21.0git" )

# Import target "LLVMHexagonDisassembler" for configuration "Release"
set_property(TARGET LLVMHexagonDisassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMHexagonDisassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMHexagonDesc;LLVMHexagonInfo;LLVMMC;LLVMMCDisassembler;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMHexagonDisassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMHexagonDisassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMHexagonDisassembler )
list(APPEND _cmake_import_check_files_for_LLVMHexagonDisassembler "${_IMPORT_PREFIX}/lib/libLLVMHexagonDisassembler.so.21.0git" )

# Import target "LLVMHexagonDesc" for configuration "Release"
set_property(TARGET LLVMHexagonDesc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMHexagonDesc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMHexagonInfo;LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMHexagonDesc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMHexagonDesc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMHexagonDesc )
list(APPEND _cmake_import_check_files_for_LLVMHexagonDesc "${_IMPORT_PREFIX}/lib/libLLVMHexagonDesc.so.21.0git" )

# Import target "LLVMHexagonInfo" for configuration "Release"
set_property(TARGET LLVMHexagonInfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMHexagonInfo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMHexagonInfo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMHexagonInfo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMHexagonInfo )
list(APPEND _cmake_import_check_files_for_LLVMHexagonInfo "${_IMPORT_PREFIX}/lib/libLLVMHexagonInfo.so.21.0git" )

# Import target "LLVMLanaiCodeGen" for configuration "Release"
set_property(TARGET LLVMLanaiCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLanaiCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMAsmPrinter;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMLanaiAsmParser;LLVMLanaiDesc;LLVMLanaiInfo;LLVMMC;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLanaiCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMLanaiCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMLanaiCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMLanaiCodeGen "${_IMPORT_PREFIX}/lib/libLLVMLanaiCodeGen.so.21.0git" )

# Import target "LLVMLanaiAsmParser" for configuration "Release"
set_property(TARGET LLVMLanaiAsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLanaiAsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGenTypes;LLVMLanaiDesc;LLVMLanaiInfo;LLVMMC;LLVMMCParser;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLanaiAsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMLanaiAsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMLanaiAsmParser )
list(APPEND _cmake_import_check_files_for_LLVMLanaiAsmParser "${_IMPORT_PREFIX}/lib/libLLVMLanaiAsmParser.so.21.0git" )

# Import target "LLVMLanaiDisassembler" for configuration "Release"
set_property(TARGET LLVMLanaiDisassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLanaiDisassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGenTypes;LLVMLanaiDesc;LLVMLanaiInfo;LLVMMC;LLVMMCDisassembler;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLanaiDisassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMLanaiDisassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMLanaiDisassembler )
list(APPEND _cmake_import_check_files_for_LLVMLanaiDisassembler "${_IMPORT_PREFIX}/lib/libLLVMLanaiDisassembler.so.21.0git" )

# Import target "LLVMLanaiDesc" for configuration "Release"
set_property(TARGET LLVMLanaiDesc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLanaiDesc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMLanaiInfo;LLVMMC;LLVMMCDisassembler;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLanaiDesc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMLanaiDesc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMLanaiDesc )
list(APPEND _cmake_import_check_files_for_LLVMLanaiDesc "${_IMPORT_PREFIX}/lib/libLLVMLanaiDesc.so.21.0git" )

# Import target "LLVMLanaiInfo" for configuration "Release"
set_property(TARGET LLVMLanaiInfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLanaiInfo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLanaiInfo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMLanaiInfo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMLanaiInfo )
list(APPEND _cmake_import_check_files_for_LLVMLanaiInfo "${_IMPORT_PREFIX}/lib/libLLVMLanaiInfo.so.21.0git" )

# Import target "LLVMLoongArchCodeGen" for configuration "Release"
set_property(TARGET LLVMLoongArchCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLoongArchCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMAsmPrinter;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMLoongArchDesc;LLVMLoongArchInfo;LLVMMC;LLVMScalarOpts;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLoongArchCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMLoongArchCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMLoongArchCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMLoongArchCodeGen "${_IMPORT_PREFIX}/lib/libLLVMLoongArchCodeGen.so.21.0git" )

# Import target "LLVMLoongArchAsmParser" for configuration "Release"
set_property(TARGET LLVMLoongArchAsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLoongArchAsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMLoongArchDesc;LLVMLoongArchInfo;LLVMMC;LLVMMCParser;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLoongArchAsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMLoongArchAsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMLoongArchAsmParser )
list(APPEND _cmake_import_check_files_for_LLVMLoongArchAsmParser "${_IMPORT_PREFIX}/lib/libLLVMLoongArchAsmParser.so.21.0git" )

# Import target "LLVMLoongArchDisassembler" for configuration "Release"
set_property(TARGET LLVMLoongArchDisassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLoongArchDisassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMLoongArchDesc;LLVMLoongArchInfo;LLVMMC;LLVMMCDisassembler;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLoongArchDisassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMLoongArchDisassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMLoongArchDisassembler )
list(APPEND _cmake_import_check_files_for_LLVMLoongArchDisassembler "${_IMPORT_PREFIX}/lib/libLLVMLoongArchDisassembler.so.21.0git" )

# Import target "LLVMLoongArchDesc" for configuration "Release"
set_property(TARGET LLVMLoongArchDesc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLoongArchDesc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMLoongArchInfo;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLoongArchDesc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMLoongArchDesc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMLoongArchDesc )
list(APPEND _cmake_import_check_files_for_LLVMLoongArchDesc "${_IMPORT_PREFIX}/lib/libLLVMLoongArchDesc.so.21.0git" )

# Import target "LLVMLoongArchInfo" for configuration "Release"
set_property(TARGET LLVMLoongArchInfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLoongArchInfo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLoongArchInfo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMLoongArchInfo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMLoongArchInfo )
list(APPEND _cmake_import_check_files_for_LLVMLoongArchInfo "${_IMPORT_PREFIX}/lib/libLLVMLoongArchInfo.so.21.0git" )

# Import target "LLVMMipsCodeGen" for configuration "Release"
set_property(TARGET LLVMMipsCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMipsCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMAsmPrinter;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMGlobalISel;LLVMMC;LLVMMipsDesc;LLVMMipsInfo;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMipsCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMMipsCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMMipsCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMMipsCodeGen "${_IMPORT_PREFIX}/lib/libLLVMMipsCodeGen.so.21.0git" )

# Import target "LLVMMipsAsmParser" for configuration "Release"
set_property(TARGET LLVMMipsAsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMipsAsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMMCParser;LLVMMipsDesc;LLVMMipsInfo;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMipsAsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMMipsAsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMMipsAsmParser )
list(APPEND _cmake_import_check_files_for_LLVMMipsAsmParser "${_IMPORT_PREFIX}/lib/libLLVMMipsAsmParser.so.21.0git" )

# Import target "LLVMMipsDisassembler" for configuration "Release"
set_property(TARGET LLVMMipsDisassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMipsDisassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMCDisassembler;LLVMMC;LLVMMipsInfo;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMipsDisassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMMipsDisassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMMipsDisassembler )
list(APPEND _cmake_import_check_files_for_LLVMMipsDisassembler "${_IMPORT_PREFIX}/lib/libLLVMMipsDisassembler.so.21.0git" )

# Import target "LLVMMipsDesc" for configuration "Release"
set_property(TARGET LLVMMipsDesc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMipsDesc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGenTypes;LLVMMC;LLVMMipsInfo;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMipsDesc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMMipsDesc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMMipsDesc )
list(APPEND _cmake_import_check_files_for_LLVMMipsDesc "${_IMPORT_PREFIX}/lib/libLLVMMipsDesc.so.21.0git" )

# Import target "LLVMMipsInfo" for configuration "Release"
set_property(TARGET LLVMMipsInfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMipsInfo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMipsInfo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMMipsInfo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMMipsInfo )
list(APPEND _cmake_import_check_files_for_LLVMMipsInfo "${_IMPORT_PREFIX}/lib/libLLVMMipsInfo.so.21.0git" )

# Import target "LLVMMSP430CodeGen" for configuration "Release"
set_property(TARGET LLVMMSP430CodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMSP430CodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAsmPrinter;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMMC;LLVMMSP430Desc;LLVMMSP430Info;LLVMSelectionDAG;LLVMSupport;LLVMTarget"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMSP430CodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMMSP430CodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMMSP430CodeGen )
list(APPEND _cmake_import_check_files_for_LLVMMSP430CodeGen "${_IMPORT_PREFIX}/lib/libLLVMMSP430CodeGen.so.21.0git" )

# Import target "LLVMMSP430Desc" for configuration "Release"
set_property(TARGET LLVMMSP430Desc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMSP430Desc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMMSP430Info;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMSP430Desc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMMSP430Desc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMMSP430Desc )
list(APPEND _cmake_import_check_files_for_LLVMMSP430Desc "${_IMPORT_PREFIX}/lib/libLLVMMSP430Desc.so.21.0git" )

# Import target "LLVMMSP430Info" for configuration "Release"
set_property(TARGET LLVMMSP430Info APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMSP430Info PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMSP430Info.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMMSP430Info.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMMSP430Info )
list(APPEND _cmake_import_check_files_for_LLVMMSP430Info "${_IMPORT_PREFIX}/lib/libLLVMMSP430Info.so.21.0git" )

# Import target "LLVMMSP430AsmParser" for configuration "Release"
set_property(TARGET LLVMMSP430AsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMSP430AsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGenTypes;LLVMMC;LLVMMCParser;LLVMMSP430Desc;LLVMMSP430Info;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMSP430AsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMMSP430AsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMMSP430AsmParser )
list(APPEND _cmake_import_check_files_for_LLVMMSP430AsmParser "${_IMPORT_PREFIX}/lib/libLLVMMSP430AsmParser.so.21.0git" )

# Import target "LLVMMSP430Disassembler" for configuration "Release"
set_property(TARGET LLVMMSP430Disassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMSP430Disassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMCDisassembler;LLVMMC;LLVMMSP430Info;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMSP430Disassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMMSP430Disassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMMSP430Disassembler )
list(APPEND _cmake_import_check_files_for_LLVMMSP430Disassembler "${_IMPORT_PREFIX}/lib/libLLVMMSP430Disassembler.so.21.0git" )

# Import target "LLVMNVPTXCodeGen" for configuration "Release"
set_property(TARGET LLVMNVPTXCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMNVPTXCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMAsmPrinter;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMipo;LLVMMC;LLVMNVPTXDesc;LLVMNVPTXInfo;LLVMScalarOpts;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils;LLVMVectorize"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMNVPTXCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMNVPTXCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMNVPTXCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMNVPTXCodeGen "${_IMPORT_PREFIX}/lib/libLLVMNVPTXCodeGen.so.21.0git" )

# Import target "LLVMNVPTXDesc" for configuration "Release"
set_property(TARGET LLVMNVPTXDesc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMNVPTXDesc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMNVPTXInfo;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMNVPTXDesc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMNVPTXDesc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMNVPTXDesc )
list(APPEND _cmake_import_check_files_for_LLVMNVPTXDesc "${_IMPORT_PREFIX}/lib/libLLVMNVPTXDesc.so.21.0git" )

# Import target "LLVMNVPTXInfo" for configuration "Release"
set_property(TARGET LLVMNVPTXInfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMNVPTXInfo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMNVPTXInfo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMNVPTXInfo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMNVPTXInfo )
list(APPEND _cmake_import_check_files_for_LLVMNVPTXInfo "${_IMPORT_PREFIX}/lib/libLLVMNVPTXInfo.so.21.0git" )

# Import target "LLVMPowerPCCodeGen" for configuration "Release"
set_property(TARGET LLVMPowerPCCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMPowerPCCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMAsmPrinter;LLVMBinaryFormat;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMGlobalISel;LLVMMC;LLVMPowerPCDesc;LLVMPowerPCInfo;LLVMScalarOpts;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMPowerPCCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMPowerPCCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMPowerPCCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMPowerPCCodeGen "${_IMPORT_PREFIX}/lib/libLLVMPowerPCCodeGen.so.21.0git" )

# Import target "LLVMPowerPCAsmParser" for configuration "Release"
set_property(TARGET LLVMPowerPCAsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMPowerPCAsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMMCParser;LLVMPowerPCDesc;LLVMPowerPCInfo;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMPowerPCAsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMPowerPCAsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMPowerPCAsmParser )
list(APPEND _cmake_import_check_files_for_LLVMPowerPCAsmParser "${_IMPORT_PREFIX}/lib/libLLVMPowerPCAsmParser.so.21.0git" )

# Import target "LLVMPowerPCDisassembler" for configuration "Release"
set_property(TARGET LLVMPowerPCDisassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMPowerPCDisassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMCDisassembler;LLVMMC;LLVMPowerPCInfo;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMPowerPCDisassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMPowerPCDisassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMPowerPCDisassembler )
list(APPEND _cmake_import_check_files_for_LLVMPowerPCDisassembler "${_IMPORT_PREFIX}/lib/libLLVMPowerPCDisassembler.so.21.0git" )

# Import target "LLVMPowerPCDesc" for configuration "Release"
set_property(TARGET LLVMPowerPCDesc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMPowerPCDesc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBinaryFormat;LLVMCodeGenTypes;LLVMMC;LLVMPowerPCInfo;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMPowerPCDesc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMPowerPCDesc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMPowerPCDesc )
list(APPEND _cmake_import_check_files_for_LLVMPowerPCDesc "${_IMPORT_PREFIX}/lib/libLLVMPowerPCDesc.so.21.0git" )

# Import target "LLVMPowerPCInfo" for configuration "Release"
set_property(TARGET LLVMPowerPCInfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMPowerPCInfo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMPowerPCInfo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMPowerPCInfo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMPowerPCInfo )
list(APPEND _cmake_import_check_files_for_LLVMPowerPCInfo "${_IMPORT_PREFIX}/lib/libLLVMPowerPCInfo.so.21.0git" )

# Import target "LLVMRISCVCodeGen" for configuration "Release"
set_property(TARGET LLVMRISCVCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMRISCVCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMAsmPrinter;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMGlobalISel;LLVMipo;LLVMMC;LLVMRISCVDesc;LLVMRISCVInfo;LLVMScalarOpts;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils;LLVMVectorize"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMRISCVCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMRISCVCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMRISCVCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMRISCVCodeGen "${_IMPORT_PREFIX}/lib/libLLVMRISCVCodeGen.so.21.0git" )

# Import target "LLVMRISCVAsmParser" for configuration "Release"
set_property(TARGET LLVMRISCVAsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMRISCVAsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMMCParser;LLVMRISCVDesc;LLVMRISCVInfo;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMRISCVAsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMRISCVAsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMRISCVAsmParser )
list(APPEND _cmake_import_check_files_for_LLVMRISCVAsmParser "${_IMPORT_PREFIX}/lib/libLLVMRISCVAsmParser.so.21.0git" )

# Import target "LLVMRISCVDisassembler" for configuration "Release"
set_property(TARGET LLVMRISCVDisassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMRISCVDisassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMMCDisassembler;LLVMRISCVDesc;LLVMRISCVInfo;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMRISCVDisassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMRISCVDisassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMRISCVDisassembler )
list(APPEND _cmake_import_check_files_for_LLVMRISCVDisassembler "${_IMPORT_PREFIX}/lib/libLLVMRISCVDisassembler.so.21.0git" )

# Import target "LLVMRISCVDesc" for configuration "Release"
set_property(TARGET LLVMRISCVDesc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMRISCVDesc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMRISCVInfo;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMRISCVDesc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMRISCVDesc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMRISCVDesc )
list(APPEND _cmake_import_check_files_for_LLVMRISCVDesc "${_IMPORT_PREFIX}/lib/libLLVMRISCVDesc.so.21.0git" )

# Import target "LLVMRISCVTargetMCA" for configuration "Release"
set_property(TARGET LLVMRISCVTargetMCA APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMRISCVTargetMCA PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGenTypes;LLVMMC;LLVMMCA;LLVMMCParser;LLVMRISCVDesc;LLVMRISCVInfo;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMRISCVTargetMCA.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMRISCVTargetMCA.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMRISCVTargetMCA )
list(APPEND _cmake_import_check_files_for_LLVMRISCVTargetMCA "${_IMPORT_PREFIX}/lib/libLLVMRISCVTargetMCA.so.21.0git" )

# Import target "LLVMRISCVInfo" for configuration "Release"
set_property(TARGET LLVMRISCVInfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMRISCVInfo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMRISCVInfo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMRISCVInfo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMRISCVInfo )
list(APPEND _cmake_import_check_files_for_LLVMRISCVInfo "${_IMPORT_PREFIX}/lib/libLLVMRISCVInfo.so.21.0git" )

# Import target "LLVMSparcCodeGen" for configuration "Release"
set_property(TARGET LLVMSparcCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSparcCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAsmPrinter;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMMC;LLVMSelectionDAG;LLVMSparcDesc;LLVMSparcInfo;LLVMSupport;LLVMTarget;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSparcCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSparcCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSparcCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMSparcCodeGen "${_IMPORT_PREFIX}/lib/libLLVMSparcCodeGen.so.21.0git" )

# Import target "LLVMSparcAsmParser" for configuration "Release"
set_property(TARGET LLVMSparcAsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSparcAsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMMCParser;LLVMSparcDesc;LLVMSparcInfo;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSparcAsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSparcAsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSparcAsmParser )
list(APPEND _cmake_import_check_files_for_LLVMSparcAsmParser "${_IMPORT_PREFIX}/lib/libLLVMSparcAsmParser.so.21.0git" )

# Import target "LLVMSparcDisassembler" for configuration "Release"
set_property(TARGET LLVMSparcDisassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSparcDisassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMCDisassembler;LLVMMC;LLVMSparcInfo;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSparcDisassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSparcDisassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSparcDisassembler )
list(APPEND _cmake_import_check_files_for_LLVMSparcDisassembler "${_IMPORT_PREFIX}/lib/libLLVMSparcDisassembler.so.21.0git" )

# Import target "LLVMSparcDesc" for configuration "Release"
set_property(TARGET LLVMSparcDesc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSparcDesc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSparcInfo;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSparcDesc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSparcDesc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSparcDesc )
list(APPEND _cmake_import_check_files_for_LLVMSparcDesc "${_IMPORT_PREFIX}/lib/libLLVMSparcDesc.so.21.0git" )

# Import target "LLVMSparcInfo" for configuration "Release"
set_property(TARGET LLVMSparcInfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSparcInfo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSparcInfo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSparcInfo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSparcInfo )
list(APPEND _cmake_import_check_files_for_LLVMSparcInfo "${_IMPORT_PREFIX}/lib/libLLVMSparcInfo.so.21.0git" )

# Import target "LLVMSPIRVCodeGen" for configuration "Release"
set_property(TARGET LLVMSPIRVCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSPIRVCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMAsmPrinter;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMDemangle;LLVMGlobalISel;LLVMSPIRVAnalysis;LLVMMC;LLVMSPIRVDesc;LLVMSPIRVInfo;LLVMScalarOpts;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSPIRVCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSPIRVCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSPIRVCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMSPIRVCodeGen "${_IMPORT_PREFIX}/lib/libLLVMSPIRVCodeGen.so.21.0git" )

# Import target "LLVMSPIRVDesc" for configuration "Release"
set_property(TARGET LLVMSPIRVDesc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSPIRVDesc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGenTypes;LLVMMC;LLVMSPIRVInfo;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSPIRVDesc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSPIRVDesc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSPIRVDesc )
list(APPEND _cmake_import_check_files_for_LLVMSPIRVDesc "${_IMPORT_PREFIX}/lib/libLLVMSPIRVDesc.so.21.0git" )

# Import target "LLVMSPIRVInfo" for configuration "Release"
set_property(TARGET LLVMSPIRVInfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSPIRVInfo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSPIRVInfo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSPIRVInfo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSPIRVInfo )
list(APPEND _cmake_import_check_files_for_LLVMSPIRVInfo "${_IMPORT_PREFIX}/lib/libLLVMSPIRVInfo.so.21.0git" )

# Import target "LLVMSPIRVAnalysis" for configuration "Release"
set_property(TARGET LLVMSPIRVAnalysis APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSPIRVAnalysis PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMSupport;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSPIRVAnalysis.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSPIRVAnalysis.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSPIRVAnalysis )
list(APPEND _cmake_import_check_files_for_LLVMSPIRVAnalysis "${_IMPORT_PREFIX}/lib/libLLVMSPIRVAnalysis.so.21.0git" )

# Import target "LLVMSystemZCodeGen" for configuration "Release"
set_property(TARGET LLVMSystemZCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSystemZCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMAsmPrinter;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMMC;LLVMScalarOpts;LLVMSelectionDAG;LLVMSupport;LLVMSystemZDesc;LLVMSystemZInfo;LLVMTarget;LLVMTargetParser;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSystemZCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSystemZCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSystemZCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMSystemZCodeGen "${_IMPORT_PREFIX}/lib/libLLVMSystemZCodeGen.so.21.0git" )

# Import target "LLVMSystemZAsmParser" for configuration "Release"
set_property(TARGET LLVMSystemZAsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSystemZAsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMMCParser;LLVMSupport;LLVMSystemZDesc;LLVMSystemZInfo"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSystemZAsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSystemZAsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSystemZAsmParser )
list(APPEND _cmake_import_check_files_for_LLVMSystemZAsmParser "${_IMPORT_PREFIX}/lib/libLLVMSystemZAsmParser.so.21.0git" )

# Import target "LLVMSystemZDisassembler" for configuration "Release"
set_property(TARGET LLVMSystemZDisassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSystemZDisassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMMCDisassembler;LLVMSupport;LLVMSystemZDesc;LLVMSystemZInfo"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSystemZDisassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSystemZDisassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSystemZDisassembler )
list(APPEND _cmake_import_check_files_for_LLVMSystemZDisassembler "${_IMPORT_PREFIX}/lib/libLLVMSystemZDisassembler.so.21.0git" )

# Import target "LLVMSystemZDesc" for configuration "Release"
set_property(TARGET LLVMSystemZDesc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSystemZDesc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGenTypes;LLVMMC;LLVMSupport;LLVMSystemZInfo"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSystemZDesc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSystemZDesc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSystemZDesc )
list(APPEND _cmake_import_check_files_for_LLVMSystemZDesc "${_IMPORT_PREFIX}/lib/libLLVMSystemZDesc.so.21.0git" )

# Import target "LLVMSystemZInfo" for configuration "Release"
set_property(TARGET LLVMSystemZInfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSystemZInfo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSystemZInfo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSystemZInfo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSystemZInfo )
list(APPEND _cmake_import_check_files_for_LLVMSystemZInfo "${_IMPORT_PREFIX}/lib/libLLVMSystemZInfo.so.21.0git" )

# Import target "LLVMVECodeGen" for configuration "Release"
set_property(TARGET LLVMVECodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMVECodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMAsmPrinter;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMMC;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTransformUtils;LLVMVEDesc;LLVMVEInfo"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMVECodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMVECodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMVECodeGen )
list(APPEND _cmake_import_check_files_for_LLVMVECodeGen "${_IMPORT_PREFIX}/lib/libLLVMVECodeGen.so.21.0git" )

# Import target "LLVMVEAsmParser" for configuration "Release"
set_property(TARGET LLVMVEAsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMVEAsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGenTypes;LLVMMC;LLVMMCParser;LLVMSupport;LLVMVEDesc;LLVMVEInfo"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMVEAsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMVEAsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMVEAsmParser )
list(APPEND _cmake_import_check_files_for_LLVMVEAsmParser "${_IMPORT_PREFIX}/lib/libLLVMVEAsmParser.so.21.0git" )

# Import target "LLVMVEDisassembler" for configuration "Release"
set_property(TARGET LLVMVEDisassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMVEDisassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGenTypes;LLVMMC;LLVMMCDisassembler;LLVMSupport;LLVMVEInfo"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMVEDisassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMVEDisassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMVEDisassembler )
list(APPEND _cmake_import_check_files_for_LLVMVEDisassembler "${_IMPORT_PREFIX}/lib/libLLVMVEDisassembler.so.21.0git" )

# Import target "LLVMVEInfo" for configuration "Release"
set_property(TARGET LLVMVEInfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMVEInfo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMVEInfo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMVEInfo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMVEInfo )
list(APPEND _cmake_import_check_files_for_LLVMVEInfo "${_IMPORT_PREFIX}/lib/libLLVMVEInfo.so.21.0git" )

# Import target "LLVMVEDesc" for configuration "Release"
set_property(TARGET LLVMVEDesc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMVEDesc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGenTypes;LLVMMC;LLVMSupport;LLVMTargetParser;LLVMVEInfo"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMVEDesc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMVEDesc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMVEDesc )
list(APPEND _cmake_import_check_files_for_LLVMVEDesc "${_IMPORT_PREFIX}/lib/libLLVMVEDesc.so.21.0git" )

# Import target "LLVMWebAssemblyCodeGen" for configuration "Release"
set_property(TARGET LLVMWebAssemblyCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMWebAssemblyCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMAsmPrinter;LLVMBinaryFormat;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMMC;LLVMScalarOpts;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils;LLVMWebAssemblyDesc;LLVMWebAssemblyInfo;LLVMWebAssemblyUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMWebAssemblyCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMWebAssemblyCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMWebAssemblyCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMWebAssemblyCodeGen "${_IMPORT_PREFIX}/lib/libLLVMWebAssemblyCodeGen.so.21.0git" )

# Import target "LLVMWebAssemblyAsmParser" for configuration "Release"
set_property(TARGET LLVMWebAssemblyAsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMWebAssemblyAsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGenTypes;LLVMMC;LLVMMCParser;LLVMSupport;LLVMTargetParser;LLVMWebAssemblyDesc;LLVMWebAssemblyInfo"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMWebAssemblyAsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMWebAssemblyAsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMWebAssemblyAsmParser )
list(APPEND _cmake_import_check_files_for_LLVMWebAssemblyAsmParser "${_IMPORT_PREFIX}/lib/libLLVMWebAssemblyAsmParser.so.21.0git" )

# Import target "LLVMWebAssemblyDisassembler" for configuration "Release"
set_property(TARGET LLVMWebAssemblyDisassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMWebAssemblyDisassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGenTypes;LLVMMC;LLVMMCDisassembler;LLVMSupport;LLVMWebAssemblyDesc;LLVMWebAssemblyInfo"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMWebAssemblyDisassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMWebAssemblyDisassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMWebAssemblyDisassembler )
list(APPEND _cmake_import_check_files_for_LLVMWebAssemblyDisassembler "${_IMPORT_PREFIX}/lib/libLLVMWebAssemblyDisassembler.so.21.0git" )

# Import target "LLVMWebAssemblyDesc" for configuration "Release"
set_property(TARGET LLVMWebAssemblyDesc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMWebAssemblyDesc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGenTypes;LLVMMC;LLVMSupport;LLVMTargetParser;LLVMWebAssemblyInfo"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMWebAssemblyDesc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMWebAssemblyDesc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMWebAssemblyDesc )
list(APPEND _cmake_import_check_files_for_LLVMWebAssemblyDesc "${_IMPORT_PREFIX}/lib/libLLVMWebAssemblyDesc.so.21.0git" )

# Import target "LLVMWebAssemblyInfo" for configuration "Release"
set_property(TARGET LLVMWebAssemblyInfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMWebAssemblyInfo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMWebAssemblyInfo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMWebAssemblyInfo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMWebAssemblyInfo )
list(APPEND _cmake_import_check_files_for_LLVMWebAssemblyInfo "${_IMPORT_PREFIX}/lib/libLLVMWebAssemblyInfo.so.21.0git" )

# Import target "LLVMWebAssemblyUtils" for configuration "Release"
set_property(TARGET LLVMWebAssemblyUtils APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMWebAssemblyUtils PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMMC;LLVMSupport;LLVMWebAssemblyDesc"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMWebAssemblyUtils.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMWebAssemblyUtils.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMWebAssemblyUtils )
list(APPEND _cmake_import_check_files_for_LLVMWebAssemblyUtils "${_IMPORT_PREFIX}/lib/libLLVMWebAssemblyUtils.so.21.0git" )

# Import target "LLVMX86CodeGen" for configuration "Release"
set_property(TARGET LLVMX86CodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMX86CodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMAsmPrinter;LLVMCFGuard;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMGlobalISel;LLVMIRPrinter;LLVMInstrumentation;LLVMMC;LLVMProfileData;LLVMScalarOpts;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils;LLVMX86Desc;LLVMX86Info"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMX86CodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMX86CodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMX86CodeGen )
list(APPEND _cmake_import_check_files_for_LLVMX86CodeGen "${_IMPORT_PREFIX}/lib/libLLVMX86CodeGen.so.21.0git" )

# Import target "LLVMX86AsmParser" for configuration "Release"
set_property(TARGET LLVMX86AsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMX86AsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMMCParser;LLVMSupport;LLVMX86Desc;LLVMX86Info"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMX86AsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMX86AsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMX86AsmParser )
list(APPEND _cmake_import_check_files_for_LLVMX86AsmParser "${_IMPORT_PREFIX}/lib/libLLVMX86AsmParser.so.21.0git" )

# Import target "LLVMX86Disassembler" for configuration "Release"
set_property(TARGET LLVMX86Disassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMX86Disassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMCDisassembler;LLVMSupport;LLVMX86Info"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMX86Disassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMX86Disassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMX86Disassembler )
list(APPEND _cmake_import_check_files_for_LLVMX86Disassembler "${_IMPORT_PREFIX}/lib/libLLVMX86Disassembler.so.21.0git" )

# Import target "LLVMX86TargetMCA" for configuration "Release"
set_property(TARGET LLVMX86TargetMCA APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMX86TargetMCA PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGenTypes;LLVMMC;LLVMMCA;LLVMMCParser;LLVMSupport;LLVMTargetParser;LLVMX86Desc;LLVMX86Info"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMX86TargetMCA.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMX86TargetMCA.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMX86TargetMCA )
list(APPEND _cmake_import_check_files_for_LLVMX86TargetMCA "${_IMPORT_PREFIX}/lib/libLLVMX86TargetMCA.so.21.0git" )

# Import target "LLVMX86Desc" for configuration "Release"
set_property(TARGET LLVMX86Desc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMX86Desc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBinaryFormat;LLVMCodeGenTypes;LLVMMC;LLVMMCDisassembler;LLVMSupport;LLVMTargetParser;LLVMX86Info"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMX86Desc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMX86Desc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMX86Desc )
list(APPEND _cmake_import_check_files_for_LLVMX86Desc "${_IMPORT_PREFIX}/lib/libLLVMX86Desc.so.21.0git" )

# Import target "LLVMX86Info" for configuration "Release"
set_property(TARGET LLVMX86Info APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMX86Info PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMX86Info.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMX86Info.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMX86Info )
list(APPEND _cmake_import_check_files_for_LLVMX86Info "${_IMPORT_PREFIX}/lib/libLLVMX86Info.so.21.0git" )

# Import target "LLVMXCoreCodeGen" for configuration "Release"
set_property(TARGET LLVMXCoreCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMXCoreCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMAsmPrinter;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMMC;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils;LLVMXCoreDesc;LLVMXCoreInfo"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMXCoreCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMXCoreCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMXCoreCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMXCoreCodeGen "${_IMPORT_PREFIX}/lib/libLLVMXCoreCodeGen.so.21.0git" )

# Import target "LLVMXCoreDisassembler" for configuration "Release"
set_property(TARGET LLVMXCoreDisassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMXCoreDisassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGenTypes;LLVMMC;LLVMMCDisassembler;LLVMSupport;LLVMXCoreInfo"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMXCoreDisassembler.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMXCoreDisassembler.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMXCoreDisassembler )
list(APPEND _cmake_import_check_files_for_LLVMXCoreDisassembler "${_IMPORT_PREFIX}/lib/libLLVMXCoreDisassembler.so.21.0git" )

# Import target "LLVMXCoreDesc" for configuration "Release"
set_property(TARGET LLVMXCoreDesc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMXCoreDesc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport;LLVMXCoreInfo"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMXCoreDesc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMXCoreDesc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMXCoreDesc )
list(APPEND _cmake_import_check_files_for_LLVMXCoreDesc "${_IMPORT_PREFIX}/lib/libLLVMXCoreDesc.so.21.0git" )

# Import target "LLVMXCoreInfo" for configuration "Release"
set_property(TARGET LLVMXCoreInfo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMXCoreInfo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMXCoreInfo.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMXCoreInfo.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMXCoreInfo )
list(APPEND _cmake_import_check_files_for_LLVMXCoreInfo "${_IMPORT_PREFIX}/lib/libLLVMXCoreInfo.so.21.0git" )

# Import target "LLVMSandboxIR" for configuration "Release"
set_property(TARGET LLVMSandboxIR APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSandboxIR PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMSupport;LLVMAnalysis"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSandboxIR.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMSandboxIR.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMSandboxIR )
list(APPEND _cmake_import_check_files_for_LLVMSandboxIR "${_IMPORT_PREFIX}/lib/libLLVMSandboxIR.so.21.0git" )

# Import target "LLVMAsmParser" for configuration "Release"
set_property(TARGET LLVMAsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBinaryFormat;LLVMCore;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAsmParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMAsmParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMAsmParser )
list(APPEND _cmake_import_check_files_for_LLVMAsmParser "${_IMPORT_PREFIX}/lib/libLLVMAsmParser.so.21.0git" )

# Import target "LLVMLineEditor" for configuration "Release"
set_property(TARGET LLVMLineEditor APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLineEditor PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLineEditor.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMLineEditor.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMLineEditor )
list(APPEND _cmake_import_check_files_for_LLVMLineEditor "${_IMPORT_PREFIX}/lib/libLLVMLineEditor.so.21.0git" )

# Import target "LLVMProfileData" for configuration "Release"
set_property(TARGET LLVMProfileData APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMProfileData PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBitstreamReader;LLVMCore;LLVMObject;LLVMSupport;LLVMDemangle;LLVMSymbolize;LLVMDebugInfoDWARF;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMProfileData.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMProfileData.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMProfileData )
list(APPEND _cmake_import_check_files_for_LLVMProfileData "${_IMPORT_PREFIX}/lib/libLLVMProfileData.so.21.0git" )

# Import target "LLVMCoverage" for configuration "Release"
set_property(TARGET LLVMCoverage APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMCoverage PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMObject;LLVMProfileData;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMCoverage.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMCoverage.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMCoverage )
list(APPEND _cmake_import_check_files_for_LLVMCoverage "${_IMPORT_PREFIX}/lib/libLLVMCoverage.so.21.0git" )

# Import target "LLVMPasses" for configuration "Release"
set_property(TARGET LLVMPasses APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMPasses PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAggressiveInstCombine;LLVMAnalysis;LLVMCFGuard;LLVMCodeGen;LLVMCore;LLVMCoroutines;LLVMHipStdPar;LLVMipo;LLVMInstCombine;LLVMIRPrinter;LLVMObjCARCOpts;LLVMScalarOpts;LLVMSupport;LLVMTarget;LLVMTransformUtils;LLVMVectorize;LLVMInstrumentation"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMPasses.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMPasses.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMPasses )
list(APPEND _cmake_import_check_files_for_LLVMPasses "${_IMPORT_PREFIX}/lib/libLLVMPasses.so.21.0git" )

# Import target "LLVMTargetParser" for configuration "Release"
set_property(TARGET LLVMTargetParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMTargetParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMTargetParser.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMTargetParser.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMTargetParser )
list(APPEND _cmake_import_check_files_for_LLVMTargetParser "${_IMPORT_PREFIX}/lib/libLLVMTargetParser.so.21.0git" )

# Import target "LLVMTextAPI" for configuration "Release"
set_property(TARGET LLVMTextAPI APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMTextAPI PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;LLVMBinaryFormat;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMTextAPI.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMTextAPI.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMTextAPI )
list(APPEND _cmake_import_check_files_for_LLVMTextAPI "${_IMPORT_PREFIX}/lib/libLLVMTextAPI.so.21.0git" )

# Import target "LLVMTextAPIBinaryReader" for configuration "Release"
set_property(TARGET LLVMTextAPIBinaryReader APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMTextAPIBinaryReader PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMDebugInfoDWARF;LLVMSupport;LLVMObject;LLVMTextAPI;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMTextAPIBinaryReader.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMTextAPIBinaryReader.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMTextAPIBinaryReader )
list(APPEND _cmake_import_check_files_for_LLVMTextAPIBinaryReader "${_IMPORT_PREFIX}/lib/libLLVMTextAPIBinaryReader.so.21.0git" )

# Import target "LLVMTelemetry" for configuration "Release"
set_property(TARGET LLVMTelemetry APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMTelemetry PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMTelemetry.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMTelemetry.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMTelemetry )
list(APPEND _cmake_import_check_files_for_LLVMTelemetry "${_IMPORT_PREFIX}/lib/libLLVMTelemetry.so.21.0git" )

# Import target "LLVMDlltoolDriver" for configuration "Release"
set_property(TARGET LLVMDlltoolDriver APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDlltoolDriver PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBinaryFormat;LLVMObject;LLVMOption;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDlltoolDriver.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMDlltoolDriver.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMDlltoolDriver )
list(APPEND _cmake_import_check_files_for_LLVMDlltoolDriver "${_IMPORT_PREFIX}/lib/libLLVMDlltoolDriver.so.21.0git" )

# Import target "LLVMLibDriver" for configuration "Release"
set_property(TARGET LLVMLibDriver APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLibDriver PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBinaryFormat;LLVMBitReader;LLVMObject;LLVMOption;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLibDriver.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMLibDriver.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMLibDriver )
list(APPEND _cmake_import_check_files_for_LLVMLibDriver "${_IMPORT_PREFIX}/lib/libLLVMLibDriver.so.21.0git" )

# Import target "LLVMXRay" for configuration "Release"
set_property(TARGET LLVMXRay APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMXRay PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;LLVMObject;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMXRay.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMXRay.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMXRay )
list(APPEND _cmake_import_check_files_for_LLVMXRay "${_IMPORT_PREFIX}/lib/libLLVMXRay.so.21.0git" )

# Import target "LLVMWindowsDriver" for configuration "Release"
set_property(TARGET LLVMWindowsDriver APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMWindowsDriver PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMOption;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMWindowsDriver.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMWindowsDriver.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMWindowsDriver )
list(APPEND _cmake_import_check_files_for_LLVMWindowsDriver "${_IMPORT_PREFIX}/lib/libLLVMWindowsDriver.so.21.0git" )

# Import target "LLVMWindowsManifest" for configuration "Release"
set_property(TARGET LLVMWindowsManifest APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMWindowsManifest PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMWindowsManifest.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMWindowsManifest.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMWindowsManifest )
list(APPEND _cmake_import_check_files_for_LLVMWindowsManifest "${_IMPORT_PREFIX}/lib/libLLVMWindowsManifest.so.21.0git" )

# Import target "omp" for configuration "Release"
set_property(TARGET omp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(omp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/x86_64-unknown-linux-gnu/libomp.so"
  IMPORTED_SONAME_RELEASE "libomp.so"
  )

list(APPEND _cmake_import_check_targets omp )
list(APPEND _cmake_import_check_files_for_omp "${_IMPORT_PREFIX}/lib/x86_64-unknown-linux-gnu/libomp.so" )

# Import target "LTO" for configuration "Release"
set_property(TARGET LTO APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LTO PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAArch64AsmParser;LLVMAMDGPUAsmParser;LLVMARMAsmParser;LLVMAVRAsmParser;LLVMBPFAsmParser;LLVMHexagonAsmParser;LLVMLanaiAsmParser;LLVMLoongArchAsmParser;LLVMMipsAsmParser;LLVMMSP430AsmParser;LLVMPowerPCAsmParser;LLVMRISCVAsmParser;LLVMSparcAsmParser;LLVMSystemZAsmParser;LLVMVEAsmParser;LLVMWebAssemblyAsmParser;LLVMX86AsmParser;LLVMAArch64CodeGen;LLVMAMDGPUCodeGen;LLVMARMCodeGen;LLVMAVRCodeGen;LLVMBPFCodeGen;LLVMHexagonCodeGen;LLVMLanaiCodeGen;LLVMLoongArchCodeGen;LLVMMipsCodeGen;LLVMMSP430CodeGen;LLVMNVPTXCodeGen;LLVMPowerPCCodeGen;LLVMRISCVCodeGen;LLVMSparcCodeGen;LLVMSPIRVCodeGen;LLVMSystemZCodeGen;LLVMVECodeGen;LLVMWebAssemblyCodeGen;LLVMX86CodeGen;LLVMXCoreCodeGen;LLVMAArch64Desc;LLVMAMDGPUDesc;LLVMARMDesc;LLVMAVRDesc;LLVMBPFDesc;LLVMHexagonDesc;LLVMLanaiDesc;LLVMLoongArchDesc;LLVMMipsDesc;LLVMMSP430Desc;LLVMNVPTXDesc;LLVMPowerPCDesc;LLVMRISCVDesc;LLVMSparcDesc;LLVMSPIRVDesc;LLVMSystemZDesc;LLVMVEDesc;LLVMWebAssemblyDesc;LLVMX86Desc;LLVMXCoreDesc;LLVMAArch64Disassembler;LLVMAMDGPUDisassembler;LLVMARMDisassembler;LLVMAVRDisassembler;LLVMBPFDisassembler;LLVMHexagonDisassembler;LLVMLanaiDisassembler;LLVMLoongArchDisassembler;LLVMMipsDisassembler;LLVMMSP430Disassembler;LLVMPowerPCDisassembler;LLVMRISCVDisassembler;LLVMSparcDisassembler;LLVMSystemZDisassembler;LLVMVEDisassembler;LLVMWebAssemblyDisassembler;LLVMX86Disassembler;LLVMXCoreDisassembler;LLVMAArch64Info;LLVMAMDGPUInfo;LLVMARMInfo;LLVMAVRInfo;LLVMBPFInfo;LLVMHexagonInfo;LLVMLanaiInfo;LLVMLoongArchInfo;LLVMMipsInfo;LLVMMSP430Info;LLVMNVPTXInfo;LLVMPowerPCInfo;LLVMRISCVInfo;LLVMSparcInfo;LLVMSPIRVInfo;LLVMSystemZInfo;LLVMVEInfo;LLVMWebAssemblyInfo;LLVMX86Info;LLVMXCoreInfo;LLVMBitReader;LLVMCore;LLVMCodeGen;LLVMLTO;LLVMMC;LLVMMCDisassembler;LLVMSupport;LLVMTarget;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLTO.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLTO.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LTO )
list(APPEND _cmake_import_check_files_for_LTO "${_IMPORT_PREFIX}/lib/libLTO.so.21.0git" )

# Import target "llvm-ar" for configuration "Release"
set_property(TARGET llvm-ar APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-ar PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-ar"
  )

list(APPEND _cmake_import_check_targets llvm-ar )
list(APPEND _cmake_import_check_files_for_llvm-ar "${_IMPORT_PREFIX}/bin/llvm-ar" )

# Import target "llvm-config" for configuration "Release"
set_property(TARGET llvm-config APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-config PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-config"
  )

list(APPEND _cmake_import_check_targets llvm-config )
list(APPEND _cmake_import_check_files_for_llvm-config "${_IMPORT_PREFIX}/bin/llvm-config" )

# Import target "llvm-ctxprof-util" for configuration "Release"
set_property(TARGET llvm-ctxprof-util APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-ctxprof-util PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-ctxprof-util"
  )

list(APPEND _cmake_import_check_targets llvm-ctxprof-util )
list(APPEND _cmake_import_check_files_for_llvm-ctxprof-util "${_IMPORT_PREFIX}/bin/llvm-ctxprof-util" )

# Import target "llvm-lto" for configuration "Release"
set_property(TARGET llvm-lto APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-lto PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-lto"
  )

list(APPEND _cmake_import_check_targets llvm-lto )
list(APPEND _cmake_import_check_files_for_llvm-lto "${_IMPORT_PREFIX}/bin/llvm-lto" )

# Import target "llvm-profdata" for configuration "Release"
set_property(TARGET llvm-profdata APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-profdata PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-profdata"
  )

list(APPEND _cmake_import_check_targets llvm-profdata )
list(APPEND _cmake_import_check_files_for_llvm-profdata "${_IMPORT_PREFIX}/bin/llvm-profdata" )

# Import target "bugpoint" for configuration "Release"
set_property(TARGET bugpoint APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(bugpoint PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/bugpoint"
  )

list(APPEND _cmake_import_check_targets bugpoint )
list(APPEND _cmake_import_check_files_for_bugpoint "${_IMPORT_PREFIX}/bin/bugpoint" )

# Import target "dsymutil" for configuration "Release"
set_property(TARGET dsymutil APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(dsymutil PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/dsymutil"
  )

list(APPEND _cmake_import_check_targets dsymutil )
list(APPEND _cmake_import_check_files_for_dsymutil "${_IMPORT_PREFIX}/bin/dsymutil" )

# Import target "llc" for configuration "Release"
set_property(TARGET llc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llc PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llc"
  )

list(APPEND _cmake_import_check_targets llc )
list(APPEND _cmake_import_check_files_for_llc "${_IMPORT_PREFIX}/bin/llc" )

# Import target "lli" for configuration "Release"
set_property(TARGET lli APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(lli PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/lli"
  )

list(APPEND _cmake_import_check_targets lli )
list(APPEND _cmake_import_check_files_for_lli "${_IMPORT_PREFIX}/bin/lli" )

# Import target "llvm-as" for configuration "Release"
set_property(TARGET llvm-as APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-as PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-as"
  )

list(APPEND _cmake_import_check_targets llvm-as )
list(APPEND _cmake_import_check_files_for_llvm-as "${_IMPORT_PREFIX}/bin/llvm-as" )

# Import target "llvm-bcanalyzer" for configuration "Release"
set_property(TARGET llvm-bcanalyzer APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-bcanalyzer PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-bcanalyzer"
  )

list(APPEND _cmake_import_check_targets llvm-bcanalyzer )
list(APPEND _cmake_import_check_files_for_llvm-bcanalyzer "${_IMPORT_PREFIX}/bin/llvm-bcanalyzer" )

# Import target "llvm-c-test" for configuration "Release"
set_property(TARGET llvm-c-test APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-c-test PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-c-test"
  )

list(APPEND _cmake_import_check_targets llvm-c-test )
list(APPEND _cmake_import_check_files_for_llvm-c-test "${_IMPORT_PREFIX}/bin/llvm-c-test" )

# Import target "llvm-cat" for configuration "Release"
set_property(TARGET llvm-cat APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-cat PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-cat"
  )

list(APPEND _cmake_import_check_targets llvm-cat )
list(APPEND _cmake_import_check_files_for_llvm-cat "${_IMPORT_PREFIX}/bin/llvm-cat" )

# Import target "llvm-cfi-verify" for configuration "Release"
set_property(TARGET llvm-cfi-verify APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-cfi-verify PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-cfi-verify"
  )

list(APPEND _cmake_import_check_targets llvm-cfi-verify )
list(APPEND _cmake_import_check_files_for_llvm-cfi-verify "${_IMPORT_PREFIX}/bin/llvm-cfi-verify" )

# Import target "LLVMCFIVerify" for configuration "Release"
set_property(TARGET LLVMCFIVerify APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMCFIVerify PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMDebugInfoDWARF;LLVMMC;LLVMMCParser;LLVMObject;LLVMSupport;LLVMSymbolize;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMCFIVerify.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMCFIVerify.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMCFIVerify )
list(APPEND _cmake_import_check_files_for_LLVMCFIVerify "${_IMPORT_PREFIX}/lib/libLLVMCFIVerify.so.21.0git" )

# Import target "llvm-cgdata" for configuration "Release"
set_property(TARGET llvm-cgdata APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-cgdata PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-cgdata"
  )

list(APPEND _cmake_import_check_targets llvm-cgdata )
list(APPEND _cmake_import_check_files_for_llvm-cgdata "${_IMPORT_PREFIX}/bin/llvm-cgdata" )

# Import target "llvm-cov" for configuration "Release"
set_property(TARGET llvm-cov APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-cov PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-cov"
  )

list(APPEND _cmake_import_check_targets llvm-cov )
list(APPEND _cmake_import_check_files_for_llvm-cov "${_IMPORT_PREFIX}/bin/llvm-cov" )

# Import target "llvm-cvtres" for configuration "Release"
set_property(TARGET llvm-cvtres APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-cvtres PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-cvtres"
  )

list(APPEND _cmake_import_check_targets llvm-cvtres )
list(APPEND _cmake_import_check_files_for_llvm-cvtres "${_IMPORT_PREFIX}/bin/llvm-cvtres" )

# Import target "llvm-cxxdump" for configuration "Release"
set_property(TARGET llvm-cxxdump APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-cxxdump PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-cxxdump"
  )

list(APPEND _cmake_import_check_targets llvm-cxxdump )
list(APPEND _cmake_import_check_files_for_llvm-cxxdump "${_IMPORT_PREFIX}/bin/llvm-cxxdump" )

# Import target "llvm-cxxfilt" for configuration "Release"
set_property(TARGET llvm-cxxfilt APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-cxxfilt PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-cxxfilt"
  )

list(APPEND _cmake_import_check_targets llvm-cxxfilt )
list(APPEND _cmake_import_check_files_for_llvm-cxxfilt "${_IMPORT_PREFIX}/bin/llvm-cxxfilt" )

# Import target "llvm-cxxmap" for configuration "Release"
set_property(TARGET llvm-cxxmap APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-cxxmap PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-cxxmap"
  )

list(APPEND _cmake_import_check_targets llvm-cxxmap )
list(APPEND _cmake_import_check_files_for_llvm-cxxmap "${_IMPORT_PREFIX}/bin/llvm-cxxmap" )

# Import target "llvm-debuginfo-analyzer" for configuration "Release"
set_property(TARGET llvm-debuginfo-analyzer APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-debuginfo-analyzer PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-debuginfo-analyzer"
  )

list(APPEND _cmake_import_check_targets llvm-debuginfo-analyzer )
list(APPEND _cmake_import_check_files_for_llvm-debuginfo-analyzer "${_IMPORT_PREFIX}/bin/llvm-debuginfo-analyzer" )

# Import target "llvm-debuginfod" for configuration "Release"
set_property(TARGET llvm-debuginfod APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-debuginfod PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-debuginfod"
  )

list(APPEND _cmake_import_check_targets llvm-debuginfod )
list(APPEND _cmake_import_check_files_for_llvm-debuginfod "${_IMPORT_PREFIX}/bin/llvm-debuginfod" )

# Import target "llvm-debuginfod-find" for configuration "Release"
set_property(TARGET llvm-debuginfod-find APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-debuginfod-find PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-debuginfod-find"
  )

list(APPEND _cmake_import_check_targets llvm-debuginfod-find )
list(APPEND _cmake_import_check_files_for_llvm-debuginfod-find "${_IMPORT_PREFIX}/bin/llvm-debuginfod-find" )

# Import target "llvm-diff" for configuration "Release"
set_property(TARGET llvm-diff APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-diff PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-diff"
  )

list(APPEND _cmake_import_check_targets llvm-diff )
list(APPEND _cmake_import_check_files_for_llvm-diff "${_IMPORT_PREFIX}/bin/llvm-diff" )

# Import target "LLVMDiff" for configuration "Release"
set_property(TARGET LLVMDiff APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDiff PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDiff.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMDiff.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMDiff )
list(APPEND _cmake_import_check_files_for_LLVMDiff "${_IMPORT_PREFIX}/lib/libLLVMDiff.so.21.0git" )

# Import target "llvm-dis" for configuration "Release"
set_property(TARGET llvm-dis APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-dis PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-dis"
  )

list(APPEND _cmake_import_check_targets llvm-dis )
list(APPEND _cmake_import_check_files_for_llvm-dis "${_IMPORT_PREFIX}/bin/llvm-dis" )

# Import target "llvm-dwarfdump" for configuration "Release"
set_property(TARGET llvm-dwarfdump APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-dwarfdump PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-dwarfdump"
  )

list(APPEND _cmake_import_check_targets llvm-dwarfdump )
list(APPEND _cmake_import_check_files_for_llvm-dwarfdump "${_IMPORT_PREFIX}/bin/llvm-dwarfdump" )

# Import target "llvm-dwarfutil" for configuration "Release"
set_property(TARGET llvm-dwarfutil APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-dwarfutil PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-dwarfutil"
  )

list(APPEND _cmake_import_check_targets llvm-dwarfutil )
list(APPEND _cmake_import_check_files_for_llvm-dwarfutil "${_IMPORT_PREFIX}/bin/llvm-dwarfutil" )

# Import target "llvm-dwp" for configuration "Release"
set_property(TARGET llvm-dwp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-dwp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-dwp"
  )

list(APPEND _cmake_import_check_targets llvm-dwp )
list(APPEND _cmake_import_check_files_for_llvm-dwp "${_IMPORT_PREFIX}/bin/llvm-dwp" )

# Import target "LLVMExegesisX86" for configuration "Release"
set_property(TARGET LLVMExegesisX86 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMExegesisX86 PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMExegesis;LLVMMC;LLVMSupport;LLVMTargetParser;LLVMX86CodeGen;LLVMX86AsmParser;LLVMX86Desc;LLVMX86Disassembler;LLVMX86Info"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMExegesisX86.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMExegesisX86.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMExegesisX86 )
list(APPEND _cmake_import_check_files_for_LLVMExegesisX86 "${_IMPORT_PREFIX}/lib/libLLVMExegesisX86.so.21.0git" )

# Import target "LLVMExegesisAArch64" for configuration "Release"
set_property(TARGET LLVMExegesisAArch64 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMExegesisAArch64 PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAArch64CodeGen;LLVMAArch64AsmParser;LLVMAArch64Desc;LLVMAArch64Disassembler;LLVMAArch64Info;LLVMAArch64Utils;LLVMCodeGenTypes;LLVMCore;LLVMExegesis;LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMExegesisAArch64.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMExegesisAArch64.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMExegesisAArch64 )
list(APPEND _cmake_import_check_files_for_LLVMExegesisAArch64 "${_IMPORT_PREFIX}/lib/libLLVMExegesisAArch64.so.21.0git" )

# Import target "LLVMExegesisPowerPC" for configuration "Release"
set_property(TARGET LLVMExegesisPowerPC APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMExegesisPowerPC PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGenTypes;LLVMCore;LLVMExegesis;LLVMMC;LLVMPowerPCCodeGen;LLVMPowerPCAsmParser;LLVMPowerPCDesc;LLVMPowerPCDisassembler;LLVMPowerPCInfo;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMExegesisPowerPC.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMExegesisPowerPC.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMExegesisPowerPC )
list(APPEND _cmake_import_check_files_for_LLVMExegesisPowerPC "${_IMPORT_PREFIX}/lib/libLLVMExegesisPowerPC.so.21.0git" )

# Import target "LLVMExegesisMips" for configuration "Release"
set_property(TARGET LLVMExegesisMips APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMExegesisMips PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGenTypes;LLVMCore;LLVMExegesis;LLVMMC;LLVMMipsCodeGen;LLVMMipsAsmParser;LLVMMipsDesc;LLVMMipsDisassembler;LLVMMipsInfo;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMExegesisMips.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMExegesisMips.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMExegesisMips )
list(APPEND _cmake_import_check_files_for_LLVMExegesisMips "${_IMPORT_PREFIX}/lib/libLLVMExegesisMips.so.21.0git" )

# Import target "LLVMExegesisRISCV" for configuration "Release"
set_property(TARGET LLVMExegesisRISCV APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMExegesisRISCV PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGen;LLVMRISCVCodeGen;LLVMRISCVAsmParser;LLVMRISCVDesc;LLVMRISCVDisassembler;LLVMRISCVInfo;LLVMExegesis;LLVMCore;LLVMSupport;LLVMMC;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMExegesisRISCV.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMExegesisRISCV.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMExegesisRISCV )
list(APPEND _cmake_import_check_files_for_LLVMExegesisRISCV "${_IMPORT_PREFIX}/lib/libLLVMExegesisRISCV.so.21.0git" )

# Import target "LLVMExegesis" for configuration "Release"
set_property(TARGET LLVMExegesis APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMExegesis PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCodeGen;LLVMCodeGenTypes;LLVMCore;LLVMExecutionEngine;LLVMGlobalISel;LLVMMC;LLVMMCA;LLVMMCDisassembler;LLVMMCParser;LLVMObject;LLVMObjectYAML;LLVMOrcJIT;LLVMRuntimeDyld;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMExegesis.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMExegesis.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMExegesis )
list(APPEND _cmake_import_check_files_for_LLVMExegesis "${_IMPORT_PREFIX}/lib/libLLVMExegesis.so.21.0git" )

# Import target "llvm-exegesis" for configuration "Release"
set_property(TARGET llvm-exegesis APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-exegesis PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-exegesis"
  )

list(APPEND _cmake_import_check_targets llvm-exegesis )
list(APPEND _cmake_import_check_files_for_llvm-exegesis "${_IMPORT_PREFIX}/bin/llvm-exegesis" )

# Import target "llvm-extract" for configuration "Release"
set_property(TARGET llvm-extract APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-extract PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-extract"
  )

list(APPEND _cmake_import_check_targets llvm-extract )
list(APPEND _cmake_import_check_files_for_llvm-extract "${_IMPORT_PREFIX}/bin/llvm-extract" )

# Import target "llvm-gsymutil" for configuration "Release"
set_property(TARGET llvm-gsymutil APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-gsymutil PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-gsymutil"
  )

list(APPEND _cmake_import_check_targets llvm-gsymutil )
list(APPEND _cmake_import_check_files_for_llvm-gsymutil "${_IMPORT_PREFIX}/bin/llvm-gsymutil" )

# Import target "llvm-ifs" for configuration "Release"
set_property(TARGET llvm-ifs APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-ifs PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-ifs"
  )

list(APPEND _cmake_import_check_targets llvm-ifs )
list(APPEND _cmake_import_check_files_for_llvm-ifs "${_IMPORT_PREFIX}/bin/llvm-ifs" )

# Import target "llvm-jitlink" for configuration "Release"
set_property(TARGET llvm-jitlink APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-jitlink PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-jitlink"
  )

list(APPEND _cmake_import_check_targets llvm-jitlink )
list(APPEND _cmake_import_check_files_for_llvm-jitlink "${_IMPORT_PREFIX}/bin/llvm-jitlink" )

# Import target "llvm-libtool-darwin" for configuration "Release"
set_property(TARGET llvm-libtool-darwin APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-libtool-darwin PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-libtool-darwin"
  )

list(APPEND _cmake_import_check_targets llvm-libtool-darwin )
list(APPEND _cmake_import_check_files_for_llvm-libtool-darwin "${_IMPORT_PREFIX}/bin/llvm-libtool-darwin" )

# Import target "llvm-link" for configuration "Release"
set_property(TARGET llvm-link APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-link PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-link"
  )

list(APPEND _cmake_import_check_targets llvm-link )
list(APPEND _cmake_import_check_files_for_llvm-link "${_IMPORT_PREFIX}/bin/llvm-link" )

# Import target "llvm-lipo" for configuration "Release"
set_property(TARGET llvm-lipo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-lipo PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-lipo"
  )

list(APPEND _cmake_import_check_targets llvm-lipo )
list(APPEND _cmake_import_check_files_for_llvm-lipo "${_IMPORT_PREFIX}/bin/llvm-lipo" )

# Import target "llvm-lto2" for configuration "Release"
set_property(TARGET llvm-lto2 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-lto2 PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-lto2"
  )

list(APPEND _cmake_import_check_targets llvm-lto2 )
list(APPEND _cmake_import_check_files_for_llvm-lto2 "${_IMPORT_PREFIX}/bin/llvm-lto2" )

# Import target "llvm-mc" for configuration "Release"
set_property(TARGET llvm-mc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-mc PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-mc"
  )

list(APPEND _cmake_import_check_targets llvm-mc )
list(APPEND _cmake_import_check_files_for_llvm-mc "${_IMPORT_PREFIX}/bin/llvm-mc" )

# Import target "llvm-mca" for configuration "Release"
set_property(TARGET llvm-mca APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-mca PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-mca"
  )

list(APPEND _cmake_import_check_targets llvm-mca )
list(APPEND _cmake_import_check_files_for_llvm-mca "${_IMPORT_PREFIX}/bin/llvm-mca" )

# Import target "llvm-ml" for configuration "Release"
set_property(TARGET llvm-ml APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-ml PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-ml"
  )

list(APPEND _cmake_import_check_targets llvm-ml )
list(APPEND _cmake_import_check_files_for_llvm-ml "${_IMPORT_PREFIX}/bin/llvm-ml" )

# Import target "llvm-modextract" for configuration "Release"
set_property(TARGET llvm-modextract APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-modextract PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-modextract"
  )

list(APPEND _cmake_import_check_targets llvm-modextract )
list(APPEND _cmake_import_check_files_for_llvm-modextract "${_IMPORT_PREFIX}/bin/llvm-modextract" )

# Import target "llvm-mt" for configuration "Release"
set_property(TARGET llvm-mt APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-mt PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-mt"
  )

list(APPEND _cmake_import_check_targets llvm-mt )
list(APPEND _cmake_import_check_files_for_llvm-mt "${_IMPORT_PREFIX}/bin/llvm-mt" )

# Import target "llvm-nm" for configuration "Release"
set_property(TARGET llvm-nm APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-nm PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-nm"
  )

list(APPEND _cmake_import_check_targets llvm-nm )
list(APPEND _cmake_import_check_files_for_llvm-nm "${_IMPORT_PREFIX}/bin/llvm-nm" )

# Import target "llvm-objcopy" for configuration "Release"
set_property(TARGET llvm-objcopy APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-objcopy PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-objcopy"
  )

list(APPEND _cmake_import_check_targets llvm-objcopy )
list(APPEND _cmake_import_check_files_for_llvm-objcopy "${_IMPORT_PREFIX}/bin/llvm-objcopy" )

# Import target "llvm-objdump" for configuration "Release"
set_property(TARGET llvm-objdump APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-objdump PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-objdump"
  )

list(APPEND _cmake_import_check_targets llvm-objdump )
list(APPEND _cmake_import_check_files_for_llvm-objdump "${_IMPORT_PREFIX}/bin/llvm-objdump" )

# Import target "llvm-opt-report" for configuration "Release"
set_property(TARGET llvm-opt-report APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-opt-report PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-opt-report"
  )

list(APPEND _cmake_import_check_targets llvm-opt-report )
list(APPEND _cmake_import_check_files_for_llvm-opt-report "${_IMPORT_PREFIX}/bin/llvm-opt-report" )

# Import target "llvm-pdbutil" for configuration "Release"
set_property(TARGET llvm-pdbutil APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-pdbutil PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-pdbutil"
  )

list(APPEND _cmake_import_check_targets llvm-pdbutil )
list(APPEND _cmake_import_check_files_for_llvm-pdbutil "${_IMPORT_PREFIX}/bin/llvm-pdbutil" )

# Import target "llvm-profgen" for configuration "Release"
set_property(TARGET llvm-profgen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-profgen PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-profgen"
  )

list(APPEND _cmake_import_check_targets llvm-profgen )
list(APPEND _cmake_import_check_files_for_llvm-profgen "${_IMPORT_PREFIX}/bin/llvm-profgen" )

# Import target "llvm-rc" for configuration "Release"
set_property(TARGET llvm-rc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-rc PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-rc"
  )

list(APPEND _cmake_import_check_targets llvm-rc )
list(APPEND _cmake_import_check_files_for_llvm-rc "${_IMPORT_PREFIX}/bin/llvm-rc" )

# Import target "llvm-readobj" for configuration "Release"
set_property(TARGET llvm-readobj APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-readobj PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-readobj"
  )

list(APPEND _cmake_import_check_targets llvm-readobj )
list(APPEND _cmake_import_check_files_for_llvm-readobj "${_IMPORT_PREFIX}/bin/llvm-readobj" )

# Import target "llvm-readtapi" for configuration "Release"
set_property(TARGET llvm-readtapi APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-readtapi PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-readtapi"
  )

list(APPEND _cmake_import_check_targets llvm-readtapi )
list(APPEND _cmake_import_check_files_for_llvm-readtapi "${_IMPORT_PREFIX}/bin/llvm-readtapi" )

# Import target "llvm-reduce" for configuration "Release"
set_property(TARGET llvm-reduce APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-reduce PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-reduce"
  )

list(APPEND _cmake_import_check_targets llvm-reduce )
list(APPEND _cmake_import_check_files_for_llvm-reduce "${_IMPORT_PREFIX}/bin/llvm-reduce" )

# Import target "llvm-remarkutil" for configuration "Release"
set_property(TARGET llvm-remarkutil APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-remarkutil PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-remarkutil"
  )

list(APPEND _cmake_import_check_targets llvm-remarkutil )
list(APPEND _cmake_import_check_files_for_llvm-remarkutil "${_IMPORT_PREFIX}/bin/llvm-remarkutil" )

# Import target "llvm-rtdyld" for configuration "Release"
set_property(TARGET llvm-rtdyld APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-rtdyld PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-rtdyld"
  )

list(APPEND _cmake_import_check_targets llvm-rtdyld )
list(APPEND _cmake_import_check_files_for_llvm-rtdyld "${_IMPORT_PREFIX}/bin/llvm-rtdyld" )

# Import target "llvm-sim" for configuration "Release"
set_property(TARGET llvm-sim APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-sim PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-sim"
  )

list(APPEND _cmake_import_check_targets llvm-sim )
list(APPEND _cmake_import_check_files_for_llvm-sim "${_IMPORT_PREFIX}/bin/llvm-sim" )

# Import target "llvm-size" for configuration "Release"
set_property(TARGET llvm-size APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-size PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-size"
  )

list(APPEND _cmake_import_check_targets llvm-size )
list(APPEND _cmake_import_check_files_for_llvm-size "${_IMPORT_PREFIX}/bin/llvm-size" )

# Import target "llvm-split" for configuration "Release"
set_property(TARGET llvm-split APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-split PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-split"
  )

list(APPEND _cmake_import_check_targets llvm-split )
list(APPEND _cmake_import_check_files_for_llvm-split "${_IMPORT_PREFIX}/bin/llvm-split" )

# Import target "llvm-stress" for configuration "Release"
set_property(TARGET llvm-stress APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-stress PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-stress"
  )

list(APPEND _cmake_import_check_targets llvm-stress )
list(APPEND _cmake_import_check_files_for_llvm-stress "${_IMPORT_PREFIX}/bin/llvm-stress" )

# Import target "llvm-strings" for configuration "Release"
set_property(TARGET llvm-strings APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-strings PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-strings"
  )

list(APPEND _cmake_import_check_targets llvm-strings )
list(APPEND _cmake_import_check_files_for_llvm-strings "${_IMPORT_PREFIX}/bin/llvm-strings" )

# Import target "llvm-symbolizer" for configuration "Release"
set_property(TARGET llvm-symbolizer APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-symbolizer PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-symbolizer"
  )

list(APPEND _cmake_import_check_targets llvm-symbolizer )
list(APPEND _cmake_import_check_files_for_llvm-symbolizer "${_IMPORT_PREFIX}/bin/llvm-symbolizer" )

# Import target "llvm-tli-checker" for configuration "Release"
set_property(TARGET llvm-tli-checker APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-tli-checker PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-tli-checker"
  )

list(APPEND _cmake_import_check_targets llvm-tli-checker )
list(APPEND _cmake_import_check_files_for_llvm-tli-checker "${_IMPORT_PREFIX}/bin/llvm-tli-checker" )

# Import target "llvm-undname" for configuration "Release"
set_property(TARGET llvm-undname APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-undname PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-undname"
  )

list(APPEND _cmake_import_check_targets llvm-undname )
list(APPEND _cmake_import_check_files_for_llvm-undname "${_IMPORT_PREFIX}/bin/llvm-undname" )

# Import target "llvm-xray" for configuration "Release"
set_property(TARGET llvm-xray APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-xray PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-xray"
  )

list(APPEND _cmake_import_check_targets llvm-xray )
list(APPEND _cmake_import_check_files_for_llvm-xray "${_IMPORT_PREFIX}/bin/llvm-xray" )

# Import target "LLVMOptDriver" for configuration "Release"
set_property(TARGET LLVMOptDriver APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMOptDriver PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAArch64AsmParser;LLVMAMDGPUAsmParser;LLVMARMAsmParser;LLVMAVRAsmParser;LLVMBPFAsmParser;LLVMHexagonAsmParser;LLVMLanaiAsmParser;LLVMLoongArchAsmParser;LLVMMipsAsmParser;LLVMMSP430AsmParser;LLVMPowerPCAsmParser;LLVMRISCVAsmParser;LLVMSparcAsmParser;LLVMSystemZAsmParser;LLVMVEAsmParser;LLVMWebAssemblyAsmParser;LLVMX86AsmParser;LLVMAArch64CodeGen;LLVMAMDGPUCodeGen;LLVMARMCodeGen;LLVMAVRCodeGen;LLVMBPFCodeGen;LLVMHexagonCodeGen;LLVMLanaiCodeGen;LLVMLoongArchCodeGen;LLVMMipsCodeGen;LLVMMSP430CodeGen;LLVMNVPTXCodeGen;LLVMPowerPCCodeGen;LLVMRISCVCodeGen;LLVMSparcCodeGen;LLVMSPIRVCodeGen;LLVMSystemZCodeGen;LLVMVECodeGen;LLVMWebAssemblyCodeGen;LLVMX86CodeGen;LLVMXCoreCodeGen;LLVMAArch64Desc;LLVMAMDGPUDesc;LLVMARMDesc;LLVMAVRDesc;LLVMBPFDesc;LLVMHexagonDesc;LLVMLanaiDesc;LLVMLoongArchDesc;LLVMMipsDesc;LLVMMSP430Desc;LLVMNVPTXDesc;LLVMPowerPCDesc;LLVMRISCVDesc;LLVMSparcDesc;LLVMSPIRVDesc;LLVMSystemZDesc;LLVMVEDesc;LLVMWebAssemblyDesc;LLVMX86Desc;LLVMXCoreDesc;LLVMAArch64Info;LLVMAMDGPUInfo;LLVMARMInfo;LLVMAVRInfo;LLVMBPFInfo;LLVMHexagonInfo;LLVMLanaiInfo;LLVMLoongArchInfo;LLVMMipsInfo;LLVMMSP430Info;LLVMNVPTXInfo;LLVMPowerPCInfo;LLVMRISCVInfo;LLVMSparcInfo;LLVMSPIRVInfo;LLVMSystemZInfo;LLVMVEInfo;LLVMWebAssemblyInfo;LLVMX86Info;LLVMXCoreInfo;LLVMAggressiveInstCombine;LLVMAnalysis;LLVMAsmParser;LLVMBitWriter;LLVMCFGuard;LLVMCodeGen;LLVMCore;LLVMCoroutines;LLVMExtensions;LLVMipo;LLVMIRReader;LLVMIRPrinter;LLVMInstCombine;LLVMInstrumentation;LLVMMC;LLVMObjCARCOpts;LLVMRemarks;LLVMScalarOpts;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils;LLVMVectorize;LLVMPasses"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMOptDriver.so.21.0git"
  IMPORTED_SONAME_RELEASE "libLLVMOptDriver.so.21.0git"
  )

list(APPEND _cmake_import_check_targets LLVMOptDriver )
list(APPEND _cmake_import_check_files_for_LLVMOptDriver "${_IMPORT_PREFIX}/lib/libLLVMOptDriver.so.21.0git" )

# Import target "opt" for configuration "Release"
set_property(TARGET opt APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opt PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opt"
  )

list(APPEND _cmake_import_check_targets opt )
list(APPEND _cmake_import_check_files_for_opt "${_IMPORT_PREFIX}/bin/opt" )

# Import target "reduce-chunk-list" for configuration "Release"
set_property(TARGET reduce-chunk-list APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(reduce-chunk-list PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/reduce-chunk-list"
  )

list(APPEND _cmake_import_check_targets reduce-chunk-list )
list(APPEND _cmake_import_check_files_for_reduce-chunk-list "${_IMPORT_PREFIX}/bin/reduce-chunk-list" )

# Import target "Remarks" for configuration "Release"
set_property(TARGET Remarks APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Remarks PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMRemarks"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libRemarks.so.21.0git"
  IMPORTED_SONAME_RELEASE "libRemarks.so.21.0git"
  )

list(APPEND _cmake_import_check_targets Remarks )
list(APPEND _cmake_import_check_files_for_Remarks "${_IMPORT_PREFIX}/lib/libRemarks.so.21.0git" )

# Import target "sancov" for configuration "Release"
set_property(TARGET sancov APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(sancov PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/sancov"
  )

list(APPEND _cmake_import_check_targets sancov )
list(APPEND _cmake_import_check_files_for_sancov "${_IMPORT_PREFIX}/bin/sancov" )

# Import target "sanstats" for configuration "Release"
set_property(TARGET sanstats APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(sanstats PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/sanstats"
  )

list(APPEND _cmake_import_check_targets sanstats )
list(APPEND _cmake_import_check_files_for_sanstats "${_IMPORT_PREFIX}/bin/sanstats" )

# Import target "verify-uselistorder" for configuration "Release"
set_property(TARGET verify-uselistorder APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(verify-uselistorder PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/verify-uselistorder"
  )

list(APPEND _cmake_import_check_targets verify-uselistorder )
list(APPEND _cmake_import_check_files_for_verify-uselistorder "${_IMPORT_PREFIX}/bin/verify-uselistorder" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
