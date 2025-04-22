#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "clang-tblgen" for configuration "Release"
set_property(TARGET clang-tblgen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-tblgen PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-tblgen"
  )

list(APPEND _cmake_import_check_targets clang-tblgen )
list(APPEND _cmake_import_check_files_for_clang-tblgen "${_IMPORT_PREFIX}/bin/clang-tblgen" )

# Import target "clangBasic" for configuration "Release"
set_property(TARGET clangBasic APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangBasic PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;LLVMTargetParser;LLVMFrontendOpenMP"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangBasic.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangBasic.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangBasic )
list(APPEND _cmake_import_check_files_for_clangBasic "${_IMPORT_PREFIX}/lib/libclangBasic.so.21.0git" )

# Import target "clangAPINotes" for configuration "Release"
set_property(TARGET clangAPINotes APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangAPINotes PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangBasic;LLVMBitReader;LLVMBitstreamReader;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangAPINotes.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangAPINotes.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangAPINotes )
list(APPEND _cmake_import_check_files_for_clangAPINotes "${_IMPORT_PREFIX}/lib/libclangAPINotes.so.21.0git" )

# Import target "clangLex" for configuration "Release"
set_property(TARGET clangLex APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangLex PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangBasic;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangLex.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangLex.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangLex )
list(APPEND _cmake_import_check_files_for_clangLex "${_IMPORT_PREFIX}/lib/libclangLex.so.21.0git" )

# Import target "clangParse" for configuration "Release"
set_property(TARGET clangParse APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangParse PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangBasic;clangLex;clangSema;LLVMFrontendOpenMP;LLVMMC;LLVMMCParser;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangParse.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangParse.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangParse )
list(APPEND _cmake_import_check_files_for_clangParse "${_IMPORT_PREFIX}/lib/libclangParse.so.21.0git" )

# Import target "clangAST" for configuration "Release"
set_property(TARGET clangAST APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangAST PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangBasic;clangLex;LLVMBinaryFormat;LLVMCore;LLVMFrontendOpenMP;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangAST.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangAST.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangAST )
list(APPEND _cmake_import_check_files_for_clangAST "${_IMPORT_PREFIX}/lib/libclangAST.so.21.0git" )

# Import target "clangDynamicASTMatchers" for configuration "Release"
set_property(TARGET clangDynamicASTMatchers APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangDynamicASTMatchers PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangASTMatchers;clangBasic;LLVMFrontendOpenMP;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangDynamicASTMatchers.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangDynamicASTMatchers.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangDynamicASTMatchers )
list(APPEND _cmake_import_check_files_for_clangDynamicASTMatchers "${_IMPORT_PREFIX}/lib/libclangDynamicASTMatchers.so.21.0git" )

# Import target "clangASTMatchers" for configuration "Release"
set_property(TARGET clangASTMatchers APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangASTMatchers PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangBasic;clangLex;LLVMFrontendOpenMP;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangASTMatchers.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangASTMatchers.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangASTMatchers )
list(APPEND _cmake_import_check_files_for_clangASTMatchers "${_IMPORT_PREFIX}/lib/libclangASTMatchers.so.21.0git" )

# Import target "clangCrossTU" for configuration "Release"
set_property(TARGET clangCrossTU APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangCrossTU PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangBasic;clangFrontend;clangIndex;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangCrossTU.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangCrossTU.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangCrossTU )
list(APPEND _cmake_import_check_files_for_clangCrossTU "${_IMPORT_PREFIX}/lib/libclangCrossTU.so.21.0git" )

# Import target "clangSema" for configuration "Release"
set_property(TARGET clangSema APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangSema PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAPINotes;clangAST;clangAnalysis;clangBasic;clangEdit;clangLex;clangSupport;LLVMCore;LLVMDemangle;LLVMFrontendHLSL;LLVMFrontendOpenMP;LLVMMC;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangSema.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangSema.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangSema )
list(APPEND _cmake_import_check_files_for_clangSema "${_IMPORT_PREFIX}/lib/libclangSema.so.21.0git" )

# Import target "clangCodeGen" for configuration "Release"
set_property(TARGET clangCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangAnalysis;clangBasic;clangFrontend;clangLex;clangSerialization;LLVMAggressiveInstCombine;LLVMAnalysis;LLVMBitReader;LLVMBitWriter;LLVMCodeGenTypes;LLVMCore;LLVMCoroutines;LLVMCoverage;LLVMDemangle;LLVMExtensions;LLVMFrontendDriver;LLVMFrontendHLSL;LLVMFrontendOpenMP;LLVMFrontendOffloading;LLVMHipStdPar;LLVMipo;LLVMIRPrinter;LLVMIRReader;LLVMInstCombine;LLVMInstrumentation;LLVMLTO;LLVMLinker;LLVMMC;LLVMObjCARCOpts;LLVMObject;LLVMPasses;LLVMProfileData;LLVMScalarOpts;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangCodeGen.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangCodeGen.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangCodeGen )
list(APPEND _cmake_import_check_files_for_clangCodeGen "${_IMPORT_PREFIX}/lib/libclangCodeGen.so.21.0git" )

# Import target "clangAnalysis" for configuration "Release"
set_property(TARGET clangAnalysis APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangAnalysis PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangASTMatchers;clangBasic;clangLex;LLVMFrontendOpenMP;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangAnalysis.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangAnalysis.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangAnalysis )
list(APPEND _cmake_import_check_files_for_clangAnalysis "${_IMPORT_PREFIX}/lib/libclangAnalysis.so.21.0git" )

# Import target "clangAnalysisFlowSensitive" for configuration "Release"
set_property(TARGET clangAnalysisFlowSensitive APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangAnalysisFlowSensitive PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAnalysis;clangAST;clangASTMatchers;clangBasic;clangLex;LLVMFrontendOpenMP;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangAnalysisFlowSensitive.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangAnalysisFlowSensitive.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangAnalysisFlowSensitive )
list(APPEND _cmake_import_check_files_for_clangAnalysisFlowSensitive "${_IMPORT_PREFIX}/lib/libclangAnalysisFlowSensitive.so.21.0git" )

# Import target "clangAnalysisFlowSensitiveModels" for configuration "Release"
set_property(TARGET clangAnalysisFlowSensitiveModels APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangAnalysisFlowSensitiveModels PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAnalysis;clangAnalysisFlowSensitive;clangAST;clangASTMatchers;clangBasic;LLVMFrontendOpenMP;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangAnalysisFlowSensitiveModels.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangAnalysisFlowSensitiveModels.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangAnalysisFlowSensitiveModels )
list(APPEND _cmake_import_check_files_for_clangAnalysisFlowSensitiveModels "${_IMPORT_PREFIX}/lib/libclangAnalysisFlowSensitiveModels.so.21.0git" )

# Import target "clangEdit" for configuration "Release"
set_property(TARGET clangEdit APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangEdit PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangBasic;clangLex;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangEdit.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangEdit.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangEdit )
list(APPEND _cmake_import_check_files_for_clangEdit "${_IMPORT_PREFIX}/lib/libclangEdit.so.21.0git" )

# Import target "clangExtractAPI" for configuration "Release"
set_property(TARGET clangExtractAPI APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangExtractAPI PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangBasic;clangFrontend;clangIndex;clangInstallAPI;clangLex;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangExtractAPI.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangExtractAPI.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangExtractAPI )
list(APPEND _cmake_import_check_files_for_clangExtractAPI "${_IMPORT_PREFIX}/lib/libclangExtractAPI.so.21.0git" )

# Import target "clangRewrite" for configuration "Release"
set_property(TARGET clangRewrite APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangRewrite PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangBasic;clangLex;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangRewrite.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangRewrite.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangRewrite )
list(APPEND _cmake_import_check_files_for_clangRewrite "${_IMPORT_PREFIX}/lib/libclangRewrite.so.21.0git" )

# Import target "clangDriver" for configuration "Release"
set_property(TARGET clangDriver APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangDriver PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangBasic;LLVMBinaryFormat;LLVMMC;LLVMObject;LLVMOption;LLVMProfileData;LLVMSupport;LLVMTargetParser;LLVMWindowsDriver"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangDriver.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangDriver.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangDriver )
list(APPEND _cmake_import_check_files_for_clangDriver "${_IMPORT_PREFIX}/lib/libclangDriver.so.21.0git" )

# Import target "clangSerialization" for configuration "Release"
set_property(TARGET clangSerialization APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangSerialization PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangBasic;clangLex;clangSema;LLVMBitReader;LLVMBitstreamReader;LLVMObject;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangSerialization.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangSerialization.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangSerialization )
list(APPEND _cmake_import_check_files_for_clangSerialization "${_IMPORT_PREFIX}/lib/libclangSerialization.so.21.0git" )

# Import target "clangRewriteFrontend" for configuration "Release"
set_property(TARGET clangRewriteFrontend APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangRewriteFrontend PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangBasic;clangEdit;clangFrontend;clangLex;clangRewrite;clangSerialization;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangRewriteFrontend.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangRewriteFrontend.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangRewriteFrontend )
list(APPEND _cmake_import_check_files_for_clangRewriteFrontend "${_IMPORT_PREFIX}/lib/libclangRewriteFrontend.so.21.0git" )

# Import target "clangFrontend" for configuration "Release"
set_property(TARGET clangFrontend APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangFrontend PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAPINotes;clangAST;clangBasic;clangDriver;clangEdit;clangLex;clangParse;clangSema;clangSerialization;LLVMBitReader;LLVMBitstreamReader;LLVMOption;LLVMProfileData;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangFrontend.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangFrontend.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangFrontend )
list(APPEND _cmake_import_check_files_for_clangFrontend "${_IMPORT_PREFIX}/lib/libclangFrontend.so.21.0git" )

# Import target "clangFrontendTool" for configuration "Release"
set_property(TARGET clangFrontendTool APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangFrontendTool PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangBasic;clangCodeGen;clangDriver;clangExtractAPI;clangFrontend;clangRewriteFrontend;clangStaticAnalyzerFrontend;LLVMOption;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangFrontendTool.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangFrontendTool.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangFrontendTool )
list(APPEND _cmake_import_check_files_for_clangFrontendTool "${_IMPORT_PREFIX}/lib/libclangFrontendTool.so.21.0git" )

# Import target "clangToolingCore" for configuration "Release"
set_property(TARGET clangToolingCore APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangToolingCore PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangBasic;clangLex;clangRewrite;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangToolingCore.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangToolingCore.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangToolingCore )
list(APPEND _cmake_import_check_files_for_clangToolingCore "${_IMPORT_PREFIX}/lib/libclangToolingCore.so.21.0git" )

# Import target "clangToolingInclusions" for configuration "Release"
set_property(TARGET clangToolingInclusions APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangToolingInclusions PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangBasic;clangLex;clangToolingCore;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangToolingInclusions.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangToolingInclusions.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangToolingInclusions )
list(APPEND _cmake_import_check_files_for_clangToolingInclusions "${_IMPORT_PREFIX}/lib/libclangToolingInclusions.so.21.0git" )

# Import target "clangToolingInclusionsStdlib" for configuration "Release"
set_property(TARGET clangToolingInclusionsStdlib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangToolingInclusionsStdlib PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangToolingInclusionsStdlib.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangToolingInclusionsStdlib.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangToolingInclusionsStdlib )
list(APPEND _cmake_import_check_files_for_clangToolingInclusionsStdlib "${_IMPORT_PREFIX}/lib/libclangToolingInclusionsStdlib.so.21.0git" )

# Import target "clangToolingRefactoring" for configuration "Release"
set_property(TARGET clangToolingRefactoring APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangToolingRefactoring PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangASTMatchers;clangBasic;clangFormat;clangIndex;clangLex;clangRewrite;clangToolingCore;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangToolingRefactoring.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangToolingRefactoring.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangToolingRefactoring )
list(APPEND _cmake_import_check_files_for_clangToolingRefactoring "${_IMPORT_PREFIX}/lib/libclangToolingRefactoring.so.21.0git" )

# Import target "clangToolingASTDiff" for configuration "Release"
set_property(TARGET clangToolingASTDiff APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangToolingASTDiff PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangBasic;clangAST;clangLex;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangToolingASTDiff.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangToolingASTDiff.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangToolingASTDiff )
list(APPEND _cmake_import_check_files_for_clangToolingASTDiff "${_IMPORT_PREFIX}/lib/libclangToolingASTDiff.so.21.0git" )

# Import target "clangToolingSyntax" for configuration "Release"
set_property(TARGET clangToolingSyntax APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangToolingSyntax PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangBasic;clangFrontend;clangLex;clangToolingCore;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangToolingSyntax.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangToolingSyntax.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangToolingSyntax )
list(APPEND _cmake_import_check_files_for_clangToolingSyntax "${_IMPORT_PREFIX}/lib/libclangToolingSyntax.so.21.0git" )

# Import target "clangDependencyScanning" for configuration "Release"
set_property(TARGET clangDependencyScanning APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangDependencyScanning PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangBasic;clangDriver;clangFrontend;clangLex;clangSerialization;clangTooling;LLVMCore;LLVMOption;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangDependencyScanning.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangDependencyScanning.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangDependencyScanning )
list(APPEND _cmake_import_check_files_for_clangDependencyScanning "${_IMPORT_PREFIX}/lib/libclangDependencyScanning.so.21.0git" )

# Import target "clangTransformer" for configuration "Release"
set_property(TARGET clangTransformer APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTransformer PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangASTMatchers;clangBasic;clangLex;clangToolingCore;clangToolingRefactoring;LLVMFrontendOpenMP;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTransformer.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTransformer.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTransformer )
list(APPEND _cmake_import_check_files_for_clangTransformer "${_IMPORT_PREFIX}/lib/libclangTransformer.so.21.0git" )

# Import target "clangTooling" for configuration "Release"
set_property(TARGET clangTooling APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTooling PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangASTMatchers;clangBasic;clangDriver;clangFormat;clangFrontend;clangLex;clangRewrite;clangSerialization;clangToolingCore;LLVMOption;LLVMFrontendOpenMP;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTooling.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTooling.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTooling )
list(APPEND _cmake_import_check_files_for_clangTooling "${_IMPORT_PREFIX}/lib/libclangTooling.so.21.0git" )

# Import target "clangDirectoryWatcher" for configuration "Release"
set_property(TARGET clangDirectoryWatcher APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangDirectoryWatcher PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangDirectoryWatcher.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangDirectoryWatcher.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangDirectoryWatcher )
list(APPEND _cmake_import_check_files_for_clangDirectoryWatcher "${_IMPORT_PREFIX}/lib/libclangDirectoryWatcher.so.21.0git" )

# Import target "clangIndex" for configuration "Release"
set_property(TARGET clangIndex APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangIndex PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangBasic;clangFormat;clangFrontend;clangLex;clangSema;clangSerialization;clangToolingCore;LLVMCore;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangIndex.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangIndex.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangIndex )
list(APPEND _cmake_import_check_files_for_clangIndex "${_IMPORT_PREFIX}/lib/libclangIndex.so.21.0git" )

# Import target "clangIndexSerialization" for configuration "Release"
set_property(TARGET clangIndexSerialization APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangIndexSerialization PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangBasic;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangIndexSerialization.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangIndexSerialization.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangIndexSerialization )
list(APPEND _cmake_import_check_files_for_clangIndexSerialization "${_IMPORT_PREFIX}/lib/libclangIndexSerialization.so.21.0git" )

# Import target "clangInstallAPI" for configuration "Release"
set_property(TARGET clangInstallAPI APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangInstallAPI PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangBasic;clangLex;LLVMSupport;LLVMTextAPI;LLVMTextAPIBinaryReader;LLVMDemangle;LLVMCore"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangInstallAPI.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangInstallAPI.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangInstallAPI )
list(APPEND _cmake_import_check_files_for_clangInstallAPI "${_IMPORT_PREFIX}/lib/libclangInstallAPI.so.21.0git" )

# Import target "clangStaticAnalyzerCore" for configuration "Release"
set_property(TARGET clangStaticAnalyzerCore APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangStaticAnalyzerCore PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangASTMatchers;clangAnalysis;clangBasic;clangCrossTU;clangFrontend;clangLex;clangRewrite;clangToolingCore;LLVMFrontendOpenMP;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangStaticAnalyzerCore.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangStaticAnalyzerCore.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangStaticAnalyzerCore )
list(APPEND _cmake_import_check_files_for_clangStaticAnalyzerCore "${_IMPORT_PREFIX}/lib/libclangStaticAnalyzerCore.so.21.0git" )

# Import target "clangStaticAnalyzerCheckers" for configuration "Release"
set_property(TARGET clangStaticAnalyzerCheckers APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangStaticAnalyzerCheckers PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangASTMatchers;clangAnalysis;clangBasic;clangLex;clangStaticAnalyzerCore;LLVMFrontendOpenMP;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangStaticAnalyzerCheckers.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangStaticAnalyzerCheckers.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangStaticAnalyzerCheckers )
list(APPEND _cmake_import_check_files_for_clangStaticAnalyzerCheckers "${_IMPORT_PREFIX}/lib/libclangStaticAnalyzerCheckers.so.21.0git" )

# Import target "clangStaticAnalyzerFrontend" for configuration "Release"
set_property(TARGET clangStaticAnalyzerFrontend APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangStaticAnalyzerFrontend PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangASTMatchers;clangAnalysis;clangBasic;clangCrossTU;clangFrontend;clangLex;clangStaticAnalyzerCheckers;clangStaticAnalyzerCore;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangStaticAnalyzerFrontend.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangStaticAnalyzerFrontend.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangStaticAnalyzerFrontend )
list(APPEND _cmake_import_check_files_for_clangStaticAnalyzerFrontend "${_IMPORT_PREFIX}/lib/libclangStaticAnalyzerFrontend.so.21.0git" )

# Import target "clangFormat" for configuration "Release"
set_property(TARGET clangFormat APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangFormat PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangBasic;clangLex;clangToolingCore;clangToolingInclusions;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangFormat.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangFormat.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangFormat )
list(APPEND _cmake_import_check_files_for_clangFormat "${_IMPORT_PREFIX}/lib/libclangFormat.so.21.0git" )

# Import target "clangInterpreter" for configuration "Release"
set_property(TARGET clangInterpreter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangInterpreter PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangAnalysis;clangBasic;clangCodeGen;clangDriver;clangFrontend;clangFrontendTool;clangLex;clangParse;clangSema;clangSerialization;LLVMCore;LLVMMC;LLVMOption;LLVMOrcJIT;LLVMOrcDebugging;LLVMOrcShared;LLVMOrcTargetProcess;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils;LLVMX86CodeGen;LLVMX86AsmParser;LLVMX86Desc;LLVMX86Disassembler;LLVMX86Info"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangInterpreter.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangInterpreter.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangInterpreter )
list(APPEND _cmake_import_check_files_for_clangInterpreter "${_IMPORT_PREFIX}/lib/libclangInterpreter.so.21.0git" )

# Import target "clangSupport" for configuration "Release"
set_property(TARGET clangSupport APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangSupport PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangSupport.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangSupport.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangSupport )
list(APPEND _cmake_import_check_files_for_clangSupport "${_IMPORT_PREFIX}/lib/libclangSupport.so.21.0git" )

# Import target "diagtool" for configuration "Release"
set_property(TARGET diagtool APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(diagtool PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/diagtool"
  )

list(APPEND _cmake_import_check_targets diagtool )
list(APPEND _cmake_import_check_files_for_diagtool "${_IMPORT_PREFIX}/bin/diagtool" )

# Import target "clang" for configuration "Release"
set_property(TARGET clang APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-21"
  )

list(APPEND _cmake_import_check_targets clang )
list(APPEND _cmake_import_check_files_for_clang "${_IMPORT_PREFIX}/bin/clang-21" )

# Import target "clang-format" for configuration "Release"
set_property(TARGET clang-format APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-format PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-format"
  )

list(APPEND _cmake_import_check_targets clang-format )
list(APPEND _cmake_import_check_files_for_clang-format "${_IMPORT_PREFIX}/bin/clang-format" )

# Import target "clangHandleCXX" for configuration "Release"
set_property(TARGET clangHandleCXX APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangHandleCXX PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangBasic;clangCodeGen;clangFrontend;clangLex;clangSerialization;clangTooling;LLVMAArch64CodeGen;LLVMAArch64AsmParser;LLVMAArch64Desc;LLVMAArch64Disassembler;LLVMAArch64Info;LLVMAArch64Utils;LLVMAMDGPUCodeGen;LLVMAMDGPUAsmParser;LLVMAMDGPUDesc;LLVMAMDGPUDisassembler;LLVMAMDGPUInfo;LLVMAMDGPUUtils;LLVMARMCodeGen;LLVMARMAsmParser;LLVMARMDesc;LLVMARMDisassembler;LLVMARMInfo;LLVMARMUtils;LLVMAVRCodeGen;LLVMAVRAsmParser;LLVMAVRDesc;LLVMAVRDisassembler;LLVMAVRInfo;LLVMBPFCodeGen;LLVMBPFAsmParser;LLVMBPFDesc;LLVMBPFDisassembler;LLVMBPFInfo;LLVMHexagonCodeGen;LLVMHexagonAsmParser;LLVMHexagonDesc;LLVMHexagonDisassembler;LLVMHexagonInfo;LLVMLanaiCodeGen;LLVMLanaiAsmParser;LLVMLanaiDesc;LLVMLanaiDisassembler;LLVMLanaiInfo;LLVMLoongArchCodeGen;LLVMLoongArchAsmParser;LLVMLoongArchDesc;LLVMLoongArchDisassembler;LLVMLoongArchInfo;LLVMMipsCodeGen;LLVMMipsAsmParser;LLVMMipsDesc;LLVMMipsDisassembler;LLVMMipsInfo;LLVMMSP430CodeGen;LLVMMSP430AsmParser;LLVMMSP430Desc;LLVMMSP430Disassembler;LLVMMSP430Info;LLVMNVPTXCodeGen;LLVMNVPTXDesc;LLVMNVPTXInfo;LLVMPowerPCCodeGen;LLVMPowerPCAsmParser;LLVMPowerPCDesc;LLVMPowerPCDisassembler;LLVMPowerPCInfo;LLVMRISCVCodeGen;LLVMRISCVAsmParser;LLVMRISCVDesc;LLVMRISCVDisassembler;LLVMRISCVInfo;LLVMSparcCodeGen;LLVMSparcAsmParser;LLVMSparcDesc;LLVMSparcDisassembler;LLVMSparcInfo;LLVMSPIRVCodeGen;LLVMSPIRVDesc;LLVMSPIRVInfo;LLVMSystemZCodeGen;LLVMSystemZAsmParser;LLVMSystemZDesc;LLVMSystemZDisassembler;LLVMSystemZInfo;LLVMVECodeGen;LLVMVEAsmParser;LLVMVEDesc;LLVMVEDisassembler;LLVMVEInfo;LLVMWebAssemblyCodeGen;LLVMWebAssemblyAsmParser;LLVMWebAssemblyDesc;LLVMWebAssemblyDisassembler;LLVMWebAssemblyInfo;LLVMWebAssemblyUtils;LLVMX86CodeGen;LLVMX86AsmParser;LLVMX86Desc;LLVMX86Disassembler;LLVMX86Info;LLVMXCoreCodeGen;LLVMXCoreDesc;LLVMXCoreDisassembler;LLVMXCoreInfo;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangHandleCXX.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangHandleCXX.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangHandleCXX )
list(APPEND _cmake_import_check_files_for_clangHandleCXX "${_IMPORT_PREFIX}/lib/libclangHandleCXX.so.21.0git" )

# Import target "clangHandleLLVM" for configuration "Release"
set_property(TARGET clangHandleLLVM APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangHandleLLVM PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCodeGen;LLVMCore;LLVMExecutionEngine;LLVMipo;LLVMIRPrinter;LLVMIRReader;LLVMMC;LLVMMCJIT;LLVMObject;LLVMPasses;LLVMRuntimeDyld;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMTargetParser;LLVMTransformUtils;LLVMX86CodeGen;LLVMX86AsmParser;LLVMX86Desc;LLVMX86Disassembler;LLVMX86Info"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangHandleLLVM.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangHandleLLVM.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangHandleLLVM )
list(APPEND _cmake_import_check_files_for_clangHandleLLVM "${_IMPORT_PREFIX}/lib/libclangHandleLLVM.so.21.0git" )

# Import target "clang-linker-wrapper" for configuration "Release"
set_property(TARGET clang-linker-wrapper APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-linker-wrapper PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-linker-wrapper"
  )

list(APPEND _cmake_import_check_targets clang-linker-wrapper )
list(APPEND _cmake_import_check_files_for_clang-linker-wrapper "${_IMPORT_PREFIX}/bin/clang-linker-wrapper" )

# Import target "clang-nvlink-wrapper" for configuration "Release"
set_property(TARGET clang-nvlink-wrapper APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-nvlink-wrapper PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-nvlink-wrapper"
  )

list(APPEND _cmake_import_check_targets clang-nvlink-wrapper )
list(APPEND _cmake_import_check_files_for_clang-nvlink-wrapper "${_IMPORT_PREFIX}/bin/clang-nvlink-wrapper" )

# Import target "clang-offload-packager" for configuration "Release"
set_property(TARGET clang-offload-packager APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-offload-packager PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-offload-packager"
  )

list(APPEND _cmake_import_check_targets clang-offload-packager )
list(APPEND _cmake_import_check_files_for_clang-offload-packager "${_IMPORT_PREFIX}/bin/clang-offload-packager" )

# Import target "clang-offload-bundler" for configuration "Release"
set_property(TARGET clang-offload-bundler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-offload-bundler PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-offload-bundler"
  )

list(APPEND _cmake_import_check_targets clang-offload-bundler )
list(APPEND _cmake_import_check_files_for_clang-offload-bundler "${_IMPORT_PREFIX}/bin/clang-offload-bundler" )

# Import target "clang-scan-deps" for configuration "Release"
set_property(TARGET clang-scan-deps APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-scan-deps PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-scan-deps"
  )

list(APPEND _cmake_import_check_targets clang-scan-deps )
list(APPEND _cmake_import_check_files_for_clang-scan-deps "${_IMPORT_PREFIX}/bin/clang-scan-deps" )

# Import target "clang-sycl-linker" for configuration "Release"
set_property(TARGET clang-sycl-linker APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-sycl-linker PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-sycl-linker"
  )

list(APPEND _cmake_import_check_targets clang-sycl-linker )
list(APPEND _cmake_import_check_files_for_clang-sycl-linker "${_IMPORT_PREFIX}/bin/clang-sycl-linker" )

# Import target "clang-installapi" for configuration "Release"
set_property(TARGET clang-installapi APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-installapi PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-installapi"
  )

list(APPEND _cmake_import_check_targets clang-installapi )
list(APPEND _cmake_import_check_files_for_clang-installapi "${_IMPORT_PREFIX}/bin/clang-installapi" )

# Import target "clang-repl" for configuration "Release"
set_property(TARGET clang-repl APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-repl PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-repl"
  )

list(APPEND _cmake_import_check_targets clang-repl )
list(APPEND _cmake_import_check_files_for_clang-repl "${_IMPORT_PREFIX}/bin/clang-repl" )

# Import target "clang-refactor" for configuration "Release"
set_property(TARGET clang-refactor APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-refactor PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-refactor"
  )

list(APPEND _cmake_import_check_targets clang-refactor )
list(APPEND _cmake_import_check_files_for_clang-refactor "${_IMPORT_PREFIX}/bin/clang-refactor" )

# Import target "clang-cpp" for configuration "Release"
set_property(TARGET clang-cpp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-cpp PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;LLVMTargetParser;LLVMFrontendOpenMP;LLVMBitReader;LLVMBitstreamReader;LLVMMC;LLVMMCParser;LLVMBinaryFormat;LLVMCore;LLVMDemangle;LLVMFrontendHLSL;LLVMAggressiveInstCombine;LLVMAnalysis;LLVMBitWriter;LLVMCodeGenTypes;LLVMCoroutines;LLVMCoverage;LLVMExtensions;LLVMFrontendDriver;LLVMFrontendOffloading;LLVMHipStdPar;LLVMipo;LLVMIRPrinter;LLVMIRReader;LLVMInstCombine;LLVMInstrumentation;LLVMLTO;LLVMLinker;LLVMObjCARCOpts;LLVMObject;LLVMPasses;LLVMProfileData;LLVMScalarOpts;LLVMTarget;LLVMTransformUtils;LLVMOption;LLVMWindowsDriver;LLVMTextAPI;LLVMTextAPIBinaryReader;LLVMOrcJIT;LLVMOrcDebugging;LLVMOrcShared;LLVMOrcTargetProcess;LLVMX86CodeGen;LLVMX86AsmParser;LLVMX86Desc;LLVMX86Disassembler;LLVMX86Info;LLVMAArch64CodeGen;LLVMAArch64AsmParser;LLVMAArch64Desc;LLVMAArch64Disassembler;LLVMAArch64Info;LLVMAArch64Utils;LLVMAMDGPUCodeGen;LLVMAMDGPUAsmParser;LLVMAMDGPUDesc;LLVMAMDGPUDisassembler;LLVMAMDGPUInfo;LLVMAMDGPUUtils;LLVMARMCodeGen;LLVMARMAsmParser;LLVMARMDesc;LLVMARMDisassembler;LLVMARMInfo;LLVMARMUtils;LLVMAVRCodeGen;LLVMAVRAsmParser;LLVMAVRDesc;LLVMAVRDisassembler;LLVMAVRInfo;LLVMBPFCodeGen;LLVMBPFAsmParser;LLVMBPFDesc;LLVMBPFDisassembler;LLVMBPFInfo;LLVMHexagonCodeGen;LLVMHexagonAsmParser;LLVMHexagonDesc;LLVMHexagonDisassembler;LLVMHexagonInfo;LLVMLanaiCodeGen;LLVMLanaiAsmParser;LLVMLanaiDesc;LLVMLanaiDisassembler;LLVMLanaiInfo;LLVMLoongArchCodeGen;LLVMLoongArchAsmParser;LLVMLoongArchDesc;LLVMLoongArchDisassembler;LLVMLoongArchInfo;LLVMMipsCodeGen;LLVMMipsAsmParser;LLVMMipsDesc;LLVMMipsDisassembler;LLVMMipsInfo;LLVMMSP430CodeGen;LLVMMSP430AsmParser;LLVMMSP430Desc;LLVMMSP430Disassembler;LLVMMSP430Info;LLVMNVPTXCodeGen;LLVMNVPTXDesc;LLVMNVPTXInfo;LLVMPowerPCCodeGen;LLVMPowerPCAsmParser;LLVMPowerPCDesc;LLVMPowerPCDisassembler;LLVMPowerPCInfo;LLVMRISCVCodeGen;LLVMRISCVAsmParser;LLVMRISCVDesc;LLVMRISCVDisassembler;LLVMRISCVInfo;LLVMSparcCodeGen;LLVMSparcAsmParser;LLVMSparcDesc;LLVMSparcDisassembler;LLVMSparcInfo;LLVMSPIRVCodeGen;LLVMSPIRVDesc;LLVMSPIRVInfo;LLVMSystemZCodeGen;LLVMSystemZAsmParser;LLVMSystemZDesc;LLVMSystemZDisassembler;LLVMSystemZInfo;LLVMVECodeGen;LLVMVEAsmParser;LLVMVEDesc;LLVMVEDisassembler;LLVMVEInfo;LLVMWebAssemblyCodeGen;LLVMWebAssemblyAsmParser;LLVMWebAssemblyDesc;LLVMWebAssemblyDisassembler;LLVMWebAssemblyInfo;LLVMWebAssemblyUtils;LLVMXCoreCodeGen;LLVMXCoreDesc;LLVMXCoreDisassembler;LLVMXCoreInfo;LLVMCodeGen;LLVMExecutionEngine;LLVMMCJIT;LLVMRuntimeDyld;LLVMSelectionDAG"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclang-cpp.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclang-cpp.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clang-cpp )
list(APPEND _cmake_import_check_files_for_clang-cpp "${_IMPORT_PREFIX}/lib/libclang-cpp.so.21.0git" )

# Import target "clang-check" for configuration "Release"
set_property(TARGET clang-check APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-check PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-check"
  )

list(APPEND _cmake_import_check_targets clang-check )
list(APPEND _cmake_import_check_files_for_clang-check "${_IMPORT_PREFIX}/bin/clang-check" )

# Import target "clang-extdef-mapping" for configuration "Release"
set_property(TARGET clang-extdef-mapping APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-extdef-mapping PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-extdef-mapping"
  )

list(APPEND _cmake_import_check_targets clang-extdef-mapping )
list(APPEND _cmake_import_check_files_for_clang-extdef-mapping "${_IMPORT_PREFIX}/bin/clang-extdef-mapping" )

# Import target "clangApplyReplacements" for configuration "Release"
set_property(TARGET clangApplyReplacements APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangApplyReplacements PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;clangAST;clangBasic;clangRewrite;clangToolingCore;clangToolingRefactoring"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangApplyReplacements.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangApplyReplacements.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangApplyReplacements )
list(APPEND _cmake_import_check_files_for_clangApplyReplacements "${_IMPORT_PREFIX}/lib/libclangApplyReplacements.so.21.0git" )

# Import target "clang-apply-replacements" for configuration "Release"
set_property(TARGET clang-apply-replacements APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-apply-replacements PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-apply-replacements"
  )

list(APPEND _cmake_import_check_targets clang-apply-replacements )
list(APPEND _cmake_import_check_files_for_clang-apply-replacements "${_IMPORT_PREFIX}/bin/clang-apply-replacements" )

# Import target "clangReorderFields" for configuration "Release"
set_property(TARGET clangReorderFields APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangReorderFields PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMFrontendOpenMP;LLVMSupport;clangAST;clangASTMatchers;clangBasic;clangIndex;clangLex;clangSerialization;clangToolingCore"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangReorderFields.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangReorderFields.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangReorderFields )
list(APPEND _cmake_import_check_files_for_clangReorderFields "${_IMPORT_PREFIX}/lib/libclangReorderFields.so.21.0git" )

# Import target "clang-reorder-fields" for configuration "Release"
set_property(TARGET clang-reorder-fields APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-reorder-fields PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-reorder-fields"
  )

list(APPEND _cmake_import_check_targets clang-reorder-fields )
list(APPEND _cmake_import_check_files_for_clang-reorder-fields "${_IMPORT_PREFIX}/bin/clang-reorder-fields" )

# Import target "modularize" for configuration "Release"
set_property(TARGET modularize APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(modularize PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/modularize"
  )

list(APPEND _cmake_import_check_targets modularize )
list(APPEND _cmake_import_check_files_for_modularize "${_IMPORT_PREFIX}/bin/modularize" )

# Import target "clangTidy" for configuration "Release"
set_property(TARGET clangTidy APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidy PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMFrontendOpenMP;LLVMSupport;clangAST;clangASTMatchers;clangAnalysis;clangBasic;clangFormat;clangFrontend;clangLex;clangRewrite;clangSerialization;clangTooling;clangToolingCore;clangStaticAnalyzerCore;clangStaticAnalyzerFrontend"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidy.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidy.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidy )
list(APPEND _cmake_import_check_files_for_clangTidy "${_IMPORT_PREFIX}/lib/libclangTidy.so.21.0git" )

# Import target "clangTidyAndroidModule" for configuration "Release"
set_property(TARGET clangTidyAndroidModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyAndroidModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyUtils;LLVMSupport;LLVMFrontendOpenMP;clangAST;clangASTMatchers;clangBasic;clangLex"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyAndroidModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyAndroidModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyAndroidModule )
list(APPEND _cmake_import_check_files_for_clangTidyAndroidModule "${_IMPORT_PREFIX}/lib/libclangTidyAndroidModule.so.21.0git" )

# Import target "clangTidyAbseilModule" for configuration "Release"
set_property(TARGET clangTidyAbseilModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyAbseilModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyUtils;LLVMSupport;LLVMFrontendOpenMP;clangAST;clangASTMatchers;clangBasic;clangLex;clangTooling;clangTransformer"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyAbseilModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyAbseilModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyAbseilModule )
list(APPEND _cmake_import_check_files_for_clangTidyAbseilModule "${_IMPORT_PREFIX}/lib/libclangTidyAbseilModule.so.21.0git" )

# Import target "clangTidyAlteraModule" for configuration "Release"
set_property(TARGET clangTidyAlteraModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyAlteraModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAnalysis;clangAST;clangASTMatchers;clangBasic;clangLex"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyAlteraModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyAlteraModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyAlteraModule )
list(APPEND _cmake_import_check_files_for_clangTidyAlteraModule "${_IMPORT_PREFIX}/lib/libclangTidyAlteraModule.so.21.0git" )

# Import target "clangTidyBoostModule" for configuration "Release"
set_property(TARGET clangTidyBoostModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyBoostModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyUtils;LLVMSupport;LLVMFrontendOpenMP;clangAST;clangASTMatchers;clangBasic;clangLex"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyBoostModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyBoostModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyBoostModule )
list(APPEND _cmake_import_check_files_for_clangTidyBoostModule "${_IMPORT_PREFIX}/lib/libclangTidyBoostModule.so.21.0git" )

# Import target "clangTidyBugproneModule" for configuration "Release"
set_property(TARGET clangTidyBugproneModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyBugproneModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyUtils;LLVMSupport;LLVMFrontendOpenMP;clangAnalysis;clangAnalysisFlowSensitive;clangAnalysisFlowSensitiveModels;clangAST;clangASTMatchers;clangBasic;clangLex;clangSema;clangTooling;clangTransformer"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyBugproneModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyBugproneModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyBugproneModule )
list(APPEND _cmake_import_check_files_for_clangTidyBugproneModule "${_IMPORT_PREFIX}/lib/libclangTidyBugproneModule.so.21.0git" )

# Import target "clangTidyCERTModule" for configuration "Release"
set_property(TARGET clangTidyCERTModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyCERTModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyBugproneModule;clangTidyConcurrencyModule;clangTidyGoogleModule;clangTidyMiscModule;clangTidyPerformanceModule;clangTidyReadabilityModule;clangTidyUtils;LLVMSupport;LLVMFrontendOpenMP;clangAST;clangASTMatchers;clangBasic;clangLex"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyCERTModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyCERTModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyCERTModule )
list(APPEND _cmake_import_check_files_for_clangTidyCERTModule "${_IMPORT_PREFIX}/lib/libclangTidyCERTModule.so.21.0git" )

# Import target "clangTidyConcurrencyModule" for configuration "Release"
set_property(TARGET clangTidyConcurrencyModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyConcurrencyModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAnalysis;clangAST;clangASTMatchers;clangBasic;clangLex;clangSerialization;clangTooling"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyConcurrencyModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyConcurrencyModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyConcurrencyModule )
list(APPEND _cmake_import_check_files_for_clangTidyConcurrencyModule "${_IMPORT_PREFIX}/lib/libclangTidyConcurrencyModule.so.21.0git" )

# Import target "clangTidyCppCoreGuidelinesModule" for configuration "Release"
set_property(TARGET clangTidyCppCoreGuidelinesModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyCppCoreGuidelinesModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyBugproneModule;clangTidyMiscModule;clangTidyModernizeModule;clangTidyPerformanceModule;clangTidyReadabilityModule;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAnalysis;clangAST;clangASTMatchers;clangBasic;clangLex;clangSerialization;clangTooling"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyCppCoreGuidelinesModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyCppCoreGuidelinesModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyCppCoreGuidelinesModule )
list(APPEND _cmake_import_check_files_for_clangTidyCppCoreGuidelinesModule "${_IMPORT_PREFIX}/lib/libclangTidyCppCoreGuidelinesModule.so.21.0git" )

# Import target "clangTidyDarwinModule" for configuration "Release"
set_property(TARGET clangTidyDarwinModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyDarwinModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAnalysis;clangAST;clangASTMatchers;clangBasic;clangLex"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyDarwinModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyDarwinModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyDarwinModule )
list(APPEND _cmake_import_check_files_for_clangTidyDarwinModule "${_IMPORT_PREFIX}/lib/libclangTidyDarwinModule.so.21.0git" )

# Import target "clangTidyFuchsiaModule" for configuration "Release"
set_property(TARGET clangTidyFuchsiaModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyFuchsiaModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyGoogleModule;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAST;clangASTMatchers;clangBasic;clangLex"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyFuchsiaModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyFuchsiaModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyFuchsiaModule )
list(APPEND _cmake_import_check_files_for_clangTidyFuchsiaModule "${_IMPORT_PREFIX}/lib/libclangTidyFuchsiaModule.so.21.0git" )

# Import target "clangTidyGoogleModule" for configuration "Release"
set_property(TARGET clangTidyGoogleModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyGoogleModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyReadabilityModule;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAST;clangASTMatchers;clangBasic;clangLex"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyGoogleModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyGoogleModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyGoogleModule )
list(APPEND _cmake_import_check_files_for_clangTidyGoogleModule "${_IMPORT_PREFIX}/lib/libclangTidyGoogleModule.so.21.0git" )

# Import target "clangTidyHICPPModule" for configuration "Release"
set_property(TARGET clangTidyHICPPModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyHICPPModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyBugproneModule;clangTidyCppCoreGuidelinesModule;clangTidyGoogleModule;clangTidyMiscModule;clangTidyModernizeModule;clangTidyPerformanceModule;clangTidyReadabilityModule;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAST;clangASTMatchers;clangBasic;clangLex;clangSerialization"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyHICPPModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyHICPPModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyHICPPModule )
list(APPEND _cmake_import_check_files_for_clangTidyHICPPModule "${_IMPORT_PREFIX}/lib/libclangTidyHICPPModule.so.21.0git" )

# Import target "clangTidyLinuxKernelModule" for configuration "Release"
set_property(TARGET clangTidyLinuxKernelModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyLinuxKernelModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAST;clangASTMatchers;clangBasic;clangLex"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyLinuxKernelModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyLinuxKernelModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyLinuxKernelModule )
list(APPEND _cmake_import_check_files_for_clangTidyLinuxKernelModule "${_IMPORT_PREFIX}/lib/libclangTidyLinuxKernelModule.so.21.0git" )

# Import target "clangTidyLLVMModule" for configuration "Release"
set_property(TARGET clangTidyLLVMModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyLLVMModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyReadabilityModule;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAST;clangASTMatchers;clangBasic;clangLex;clangTooling"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyLLVMModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyLLVMModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyLLVMModule )
list(APPEND _cmake_import_check_files_for_clangTidyLLVMModule "${_IMPORT_PREFIX}/lib/libclangTidyLLVMModule.so.21.0git" )

# Import target "clangTidyLLVMLibcModule" for configuration "Release"
set_property(TARGET clangTidyLLVMLibcModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyLLVMLibcModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyPortabilityModule;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAST;clangASTMatchers;clangBasic;clangLex;clangTooling"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyLLVMLibcModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyLLVMLibcModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyLLVMLibcModule )
list(APPEND _cmake_import_check_files_for_clangTidyLLVMLibcModule "${_IMPORT_PREFIX}/lib/libclangTidyLLVMLibcModule.so.21.0git" )

# Import target "clangTidyMiscModule" for configuration "Release"
set_property(TARGET clangTidyMiscModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyMiscModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAnalysis;clangAST;clangASTMatchers;clangBasic;clangFormat;clangLex;clangSerialization;clangTooling;clangToolingInclusions;clangToolingInclusionsStdlib;clangIncludeCleaner"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyMiscModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyMiscModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyMiscModule )
list(APPEND _cmake_import_check_files_for_clangTidyMiscModule "${_IMPORT_PREFIX}/lib/libclangTidyMiscModule.so.21.0git" )

# Import target "clangTidyModernizeModule" for configuration "Release"
set_property(TARGET clangTidyModernizeModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyModernizeModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyReadabilityModule;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAST;clangASTMatchers;clangBasic;clangLex;clangTooling"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyModernizeModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyModernizeModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyModernizeModule )
list(APPEND _cmake_import_check_files_for_clangTidyModernizeModule "${_IMPORT_PREFIX}/lib/libclangTidyModernizeModule.so.21.0git" )

# Import target "clangTidyMPIModule" for configuration "Release"
set_property(TARGET clangTidyMPIModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyMPIModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAnalysis;clangAST;clangASTMatchers;clangBasic;clangLex;clangTooling;clangStaticAnalyzerCheckers"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyMPIModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyMPIModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyMPIModule )
list(APPEND _cmake_import_check_files_for_clangTidyMPIModule "${_IMPORT_PREFIX}/lib/libclangTidyMPIModule.so.21.0git" )

# Import target "clangTidyObjCModule" for configuration "Release"
set_property(TARGET clangTidyObjCModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyObjCModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAST;clangASTMatchers;clangBasic;clangLex"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyObjCModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyObjCModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyObjCModule )
list(APPEND _cmake_import_check_files_for_clangTidyObjCModule "${_IMPORT_PREFIX}/lib/libclangTidyObjCModule.so.21.0git" )

# Import target "clangTidyOpenMPModule" for configuration "Release"
set_property(TARGET clangTidyOpenMPModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyOpenMPModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAST;clangASTMatchers;clangBasic"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyOpenMPModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyOpenMPModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyOpenMPModule )
list(APPEND _cmake_import_check_files_for_clangTidyOpenMPModule "${_IMPORT_PREFIX}/lib/libclangTidyOpenMPModule.so.21.0git" )

# Import target "clangTidyPerformanceModule" for configuration "Release"
set_property(TARGET clangTidyPerformanceModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyPerformanceModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAST;clangASTMatchers;clangAnalysis;clangBasic;clangLex"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyPerformanceModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyPerformanceModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyPerformanceModule )
list(APPEND _cmake_import_check_files_for_clangTidyPerformanceModule "${_IMPORT_PREFIX}/lib/libclangTidyPerformanceModule.so.21.0git" )

# Import target "clangTidyPortabilityModule" for configuration "Release"
set_property(TARGET clangTidyPortabilityModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyPortabilityModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;LLVMTargetParser;clangAST;clangASTMatchers;clangBasic;clangLex;clangTooling"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyPortabilityModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyPortabilityModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyPortabilityModule )
list(APPEND _cmake_import_check_files_for_clangTidyPortabilityModule "${_IMPORT_PREFIX}/lib/libclangTidyPortabilityModule.so.21.0git" )

# Import target "clangTidyReadabilityModule" for configuration "Release"
set_property(TARGET clangTidyReadabilityModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyReadabilityModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAnalysis;clangAST;clangASTMatchers;clangBasic;clangLex;clangTooling"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyReadabilityModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyReadabilityModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyReadabilityModule )
list(APPEND _cmake_import_check_files_for_clangTidyReadabilityModule "${_IMPORT_PREFIX}/lib/libclangTidyReadabilityModule.so.21.0git" )

# Import target "clangTidyZirconModule" for configuration "Release"
set_property(TARGET clangTidyZirconModule APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyZirconModule PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyUtils;LLVMFrontendOpenMP;LLVMSupport;clangAST;clangASTMatchers;clangBasic;clangLex"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyZirconModule.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyZirconModule.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyZirconModule )
list(APPEND _cmake_import_check_files_for_clangTidyZirconModule "${_IMPORT_PREFIX}/lib/libclangTidyZirconModule.so.21.0git" )

# Import target "clangTidyPlugin" for configuration "Release"
set_property(TARGET clangTidyPlugin APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyPlugin PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyAndroidModule;clangTidyAbseilModule;clangTidyAlteraModule;clangTidyBoostModule;clangTidyBugproneModule;clangTidyCERTModule;clangTidyConcurrencyModule;clangTidyCppCoreGuidelinesModule;clangTidyDarwinModule;clangTidyFuchsiaModule;clangTidyGoogleModule;clangTidyHICPPModule;clangTidyLinuxKernelModule;clangTidyLLVMModule;clangTidyLLVMLibcModule;clangTidyMiscModule;clangTidyModernizeModule;clangTidyObjCModule;clangTidyOpenMPModule;clangTidyPerformanceModule;clangTidyPortabilityModule;clangTidyReadabilityModule;clangTidyZirconModule;clangTidyMPIModule;LLVMFrontendOpenMP;LLVMSupport;clangAST;clangASTMatchers;clangBasic;clangFrontend;clangSema;clangTooling"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyPlugin.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyPlugin.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyPlugin )
list(APPEND _cmake_import_check_files_for_clangTidyPlugin "${_IMPORT_PREFIX}/lib/libclangTidyPlugin.so.21.0git" )

# Import target "clangTidyMain" for configuration "Release"
set_property(TARGET clangTidyMain APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyMain PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;clangTidyAndroidModule;clangTidyAbseilModule;clangTidyAlteraModule;clangTidyBoostModule;clangTidyBugproneModule;clangTidyCERTModule;clangTidyConcurrencyModule;clangTidyCppCoreGuidelinesModule;clangTidyDarwinModule;clangTidyFuchsiaModule;clangTidyGoogleModule;clangTidyHICPPModule;clangTidyLinuxKernelModule;clangTidyLLVMModule;clangTidyLLVMLibcModule;clangTidyMiscModule;clangTidyModernizeModule;clangTidyObjCModule;clangTidyOpenMPModule;clangTidyPerformanceModule;clangTidyPortabilityModule;clangTidyReadabilityModule;clangTidyZirconModule;clangTidyMPIModule;LLVMAArch64AsmParser;LLVMAMDGPUAsmParser;LLVMARMAsmParser;LLVMAVRAsmParser;LLVMBPFAsmParser;LLVMHexagonAsmParser;LLVMLanaiAsmParser;LLVMLoongArchAsmParser;LLVMMipsAsmParser;LLVMMSP430AsmParser;LLVMPowerPCAsmParser;LLVMRISCVAsmParser;LLVMSparcAsmParser;LLVMSystemZAsmParser;LLVMVEAsmParser;LLVMWebAssemblyAsmParser;LLVMX86AsmParser;LLVMAArch64Desc;LLVMAMDGPUDesc;LLVMARMDesc;LLVMAVRDesc;LLVMBPFDesc;LLVMHexagonDesc;LLVMLanaiDesc;LLVMLoongArchDesc;LLVMMipsDesc;LLVMMSP430Desc;LLVMNVPTXDesc;LLVMPowerPCDesc;LLVMRISCVDesc;LLVMSparcDesc;LLVMSPIRVDesc;LLVMSystemZDesc;LLVMVEDesc;LLVMWebAssemblyDesc;LLVMX86Desc;LLVMXCoreDesc;LLVMAArch64Info;LLVMAMDGPUInfo;LLVMARMInfo;LLVMAVRInfo;LLVMBPFInfo;LLVMHexagonInfo;LLVMLanaiInfo;LLVMLoongArchInfo;LLVMMipsInfo;LLVMMSP430Info;LLVMNVPTXInfo;LLVMPowerPCInfo;LLVMRISCVInfo;LLVMSparcInfo;LLVMSPIRVInfo;LLVMSystemZInfo;LLVMVEInfo;LLVMWebAssemblyInfo;LLVMX86Info;LLVMXCoreInfo;LLVMFrontendOpenMP;LLVMTargetParser;LLVMSupport;clangAST;clangASTMatchers;clangBasic;clangTooling;clangToolingCore"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyMain.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyMain.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyMain )
list(APPEND _cmake_import_check_files_for_clangTidyMain "${_IMPORT_PREFIX}/lib/libclangTidyMain.so.21.0git" )

# Import target "clang-tidy" for configuration "Release"
set_property(TARGET clang-tidy APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-tidy PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-tidy"
  )

list(APPEND _cmake_import_check_targets clang-tidy )
list(APPEND _cmake_import_check_files_for_clang-tidy "${_IMPORT_PREFIX}/bin/clang-tidy" )

# Import target "clangTidyUtils" for configuration "Release"
set_property(TARGET clangTidyUtils APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangTidyUtils PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangTidy;LLVMFrontendOpenMP;LLVMSupport;clangAST;clangASTMatchers;clangBasic;clangLex;clangSema;clangTooling;clangTransformer"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangTidyUtils.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangTidyUtils.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangTidyUtils )
list(APPEND _cmake_import_check_files_for_clangTidyUtils "${_IMPORT_PREFIX}/lib/libclangTidyUtils.so.21.0git" )

# Import target "clangChangeNamespace" for configuration "Release"
set_property(TARGET clangChangeNamespace APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangChangeNamespace PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMFrontendOpenMP;LLVMSupport;clangAST;clangASTMatchers;clangBasic;clangFormat;clangFrontend;clangLex;clangSerialization;clangTooling;clangToolingCore"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangChangeNamespace.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangChangeNamespace.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangChangeNamespace )
list(APPEND _cmake_import_check_files_for_clangChangeNamespace "${_IMPORT_PREFIX}/lib/libclangChangeNamespace.so.21.0git" )

# Import target "clang-change-namespace" for configuration "Release"
set_property(TARGET clang-change-namespace APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-change-namespace PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-change-namespace"
  )

list(APPEND _cmake_import_check_targets clang-change-namespace )
list(APPEND _cmake_import_check_files_for_clang-change-namespace "${_IMPORT_PREFIX}/bin/clang-change-namespace" )

# Import target "clangDocSupport" for configuration "Release"
set_property(TARGET clangDocSupport APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangDocSupport PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangDocSupport.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangDocSupport.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangDocSupport )
list(APPEND _cmake_import_check_files_for_clangDocSupport "${_IMPORT_PREFIX}/lib/libclangDocSupport.so.21.0git" )

# Import target "clangDoc" for configuration "Release"
set_property(TARGET clangDoc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangDoc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;LLVMBitstreamReader;LLVMFrontendOpenMP;clangDocSupport;clangAnalysis;clangAST;clangASTMatchers;clangBasic;clangFrontend;clangIndex;clangLex;clangTooling;clangToolingCore"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangDoc.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangDoc.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangDoc )
list(APPEND _cmake_import_check_files_for_clangDoc "${_IMPORT_PREFIX}/lib/libclangDoc.so.21.0git" )

# Import target "clang-doc" for configuration "Release"
set_property(TARGET clang-doc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-doc PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-doc"
  )

list(APPEND _cmake_import_check_targets clang-doc )
list(APPEND _cmake_import_check_files_for_clang-doc "${_IMPORT_PREFIX}/bin/clang-doc" )

# Import target "clangIncludeFixer" for configuration "Release"
set_property(TARGET clangIncludeFixer APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangIncludeFixer PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "findAllSymbols;LLVMSupport;clangAST;clangBasic;clangFormat;clangFrontend;clangLex;clangParse;clangSema;clangSerialization;clangTooling;clangToolingCore"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangIncludeFixer.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangIncludeFixer.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangIncludeFixer )
list(APPEND _cmake_import_check_files_for_clangIncludeFixer "${_IMPORT_PREFIX}/lib/libclangIncludeFixer.so.21.0git" )

# Import target "clangIncludeFixerPlugin" for configuration "Release"
set_property(TARGET clangIncludeFixerPlugin APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangIncludeFixerPlugin PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangBasic;clangFrontend;clangIncludeFixer;clangParse;clangSema;clangTooling;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangIncludeFixerPlugin.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangIncludeFixerPlugin.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangIncludeFixerPlugin )
list(APPEND _cmake_import_check_files_for_clangIncludeFixerPlugin "${_IMPORT_PREFIX}/lib/libclangIncludeFixerPlugin.so.21.0git" )

# Import target "clang-include-fixer" for configuration "Release"
set_property(TARGET clang-include-fixer APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-include-fixer PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-include-fixer"
  )

list(APPEND _cmake_import_check_targets clang-include-fixer )
list(APPEND _cmake_import_check_files_for_clang-include-fixer "${_IMPORT_PREFIX}/bin/clang-include-fixer" )

# Import target "findAllSymbols" for configuration "Release"
set_property(TARGET findAllSymbols APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(findAllSymbols PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;LLVMFrontendOpenMP;clangAST;clangASTMatchers;clangBasic;clangFrontend;clangLex;clangTooling"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libfindAllSymbols.so.21.0git"
  IMPORTED_SONAME_RELEASE "libfindAllSymbols.so.21.0git"
  )

list(APPEND _cmake_import_check_targets findAllSymbols )
list(APPEND _cmake_import_check_files_for_findAllSymbols "${_IMPORT_PREFIX}/lib/libfindAllSymbols.so.21.0git" )

# Import target "find-all-symbols" for configuration "Release"
set_property(TARGET find-all-symbols APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(find-all-symbols PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/find-all-symbols"
  )

list(APPEND _cmake_import_check_targets find-all-symbols )
list(APPEND _cmake_import_check_files_for_find-all-symbols "${_IMPORT_PREFIX}/bin/find-all-symbols" )

# Import target "clangMove" for configuration "Release"
set_property(TARGET clangMove APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangMove PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;LLVMFrontendOpenMP;clangAnalysis;clangAST;clangASTMatchers;clangBasic;clangFormat;clangFrontend;clangLex;clangSerialization;clangTooling;clangToolingCore"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangMove.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangMove.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangMove )
list(APPEND _cmake_import_check_files_for_clangMove "${_IMPORT_PREFIX}/lib/libclangMove.so.21.0git" )

# Import target "clang-move" for configuration "Release"
set_property(TARGET clang-move APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-move PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-move"
  )

list(APPEND _cmake_import_check_targets clang-move )
list(APPEND _cmake_import_check_files_for_clang-move "${_IMPORT_PREFIX}/bin/clang-move" )

# Import target "clangQuery" for configuration "Release"
set_property(TARGET clangQuery APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangQuery PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMLineEditor;LLVMSupport;LLVMFrontendOpenMP;clangAST;clangASTMatchers;clangBasic;clangDynamicASTMatchers;clangFrontend;clangSerialization"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangQuery.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangQuery.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangQuery )
list(APPEND _cmake_import_check_files_for_clangQuery "${_IMPORT_PREFIX}/lib/libclangQuery.so.21.0git" )

# Import target "clang-query" for configuration "Release"
set_property(TARGET clang-query APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-query PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-query"
  )

list(APPEND _cmake_import_check_targets clang-query )
list(APPEND _cmake_import_check_files_for_clang-query "${_IMPORT_PREFIX}/bin/clang-query" )

# Import target "clangIncludeCleaner" for configuration "Release"
set_property(TARGET clangIncludeCleaner APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangIncludeCleaner PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;clangAST;clangBasic;clangFormat;clangLex;clangToolingCore;clangToolingInclusions;clangToolingInclusionsStdlib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangIncludeCleaner.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangIncludeCleaner.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangIncludeCleaner )
list(APPEND _cmake_import_check_files_for_clangIncludeCleaner "${_IMPORT_PREFIX}/lib/libclangIncludeCleaner.so.21.0git" )

# Import target "clang-include-cleaner" for configuration "Release"
set_property(TARGET clang-include-cleaner APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clang-include-cleaner PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clang-include-cleaner"
  )

list(APPEND _cmake_import_check_targets clang-include-cleaner )
list(APPEND _cmake_import_check_files_for_clang-include-cleaner "${_IMPORT_PREFIX}/bin/clang-include-cleaner" )

# Import target "pp-trace" for configuration "Release"
set_property(TARGET pp-trace APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(pp-trace PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/pp-trace"
  )

list(APPEND _cmake_import_check_targets pp-trace )
list(APPEND _cmake_import_check_files_for_pp-trace "${_IMPORT_PREFIX}/bin/pp-trace" )

# Import target "clangdSupport" for configuration "Release"
set_property(TARGET clangdSupport APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangdSupport PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;clangBasic;clangLex"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangdSupport.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangdSupport.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangdSupport )
list(APPEND _cmake_import_check_files_for_clangdSupport "${_IMPORT_PREFIX}/lib/libclangdSupport.so.21.0git" )

# Import target "clangDaemon" for configuration "Release"
set_property(TARGET clangDaemon APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangDaemon PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;LLVMAArch64Info;LLVMAMDGPUInfo;LLVMARMInfo;LLVMAVRInfo;LLVMBPFInfo;LLVMHexagonInfo;LLVMLanaiInfo;LLVMLoongArchInfo;LLVMMipsInfo;LLVMMSP430Info;LLVMNVPTXInfo;LLVMPowerPCInfo;LLVMRISCVInfo;LLVMSparcInfo;LLVMSPIRVInfo;LLVMSystemZInfo;LLVMVEInfo;LLVMWebAssemblyInfo;LLVMX86Info;LLVMXCoreInfo;LLVMFrontendOpenMP;LLVMOption;LLVMTargetParser;clangAST;clangASTMatchers;clangBasic;clangDependencyScanning;clangDriver;clangFormat;clangFrontend;clangIndex;clangLex;clangSema;clangSerialization;clangTooling;clangToolingCore;clangToolingInclusions;clangToolingInclusionsStdlib;clangToolingSyntax;clangIncludeCleaner;clangTidy;clangTidyUtils;clangdSupport;clangTidyAndroidModule;clangTidyAbseilModule;clangTidyAlteraModule;clangTidyBoostModule;clangTidyBugproneModule;clangTidyCERTModule;clangTidyConcurrencyModule;clangTidyCppCoreGuidelinesModule;clangTidyDarwinModule;clangTidyFuchsiaModule;clangTidyGoogleModule;clangTidyHICPPModule;clangTidyLinuxKernelModule;clangTidyLLVMModule;clangTidyLLVMLibcModule;clangTidyMiscModule;clangTidyModernizeModule;clangTidyObjCModule;clangTidyOpenMPModule;clangTidyPerformanceModule;clangTidyPortabilityModule;clangTidyReadabilityModule;clangTidyZirconModule;clangTidyMPIModule"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangDaemon.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangDaemon.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangDaemon )
list(APPEND _cmake_import_check_files_for_clangDaemon "${_IMPORT_PREFIX}/lib/libclangDaemon.so.21.0git" )

# Import target "clangDaemonTweaks" for configuration "Release"
set_property(TARGET clangDaemonTweaks APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangDaemonTweaks PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangBasic;clangDaemon;clangdSupport;clangFormat;clangLex;clangSema;clangToolingCore;clangToolingRefactoring;clangToolingSyntax;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangDaemonTweaks.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangDaemonTweaks.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangDaemonTweaks )
list(APPEND _cmake_import_check_files_for_clangDaemonTweaks "${_IMPORT_PREFIX}/lib/libclangDaemonTweaks.so.21.0git" )

# Import target "clangdMain" for configuration "Release"
set_property(TARGET clangdMain APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangdMain PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport;LLVMAArch64Info;LLVMAMDGPUInfo;LLVMARMInfo;LLVMAVRInfo;LLVMBPFInfo;LLVMHexagonInfo;LLVMLanaiInfo;LLVMLoongArchInfo;LLVMMipsInfo;LLVMMSP430Info;LLVMNVPTXInfo;LLVMPowerPCInfo;LLVMRISCVInfo;LLVMSparcInfo;LLVMSPIRVInfo;LLVMSystemZInfo;LLVMVEInfo;LLVMWebAssemblyInfo;LLVMX86Info;LLVMXCoreInfo;LLVMFrontendOpenMP;LLVMOption;LLVMTargetParser;clangAST;clangBasic;clangFormat;clangFrontend;clangTooling;clangToolingSyntax;clangTidy;clangTidyUtils;clangDaemon;clangdRemoteIndex;clangdSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangdMain.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangdMain.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangdMain )
list(APPEND _cmake_import_check_files_for_clangdMain "${_IMPORT_PREFIX}/lib/libclangdMain.so.21.0git" )

# Import target "clangd" for configuration "Release"
set_property(TARGET clangd APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangd PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/clangd"
  )

list(APPEND _cmake_import_check_targets clangd )
list(APPEND _cmake_import_check_files_for_clangd "${_IMPORT_PREFIX}/bin/clangd" )

# Import target "clangdRemoteIndex" for configuration "Release"
set_property(TARGET clangdRemoteIndex APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(clangdRemoteIndex PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangdSupport;LLVMSupport;LLVMAArch64Info;LLVMAMDGPUInfo;LLVMARMInfo;LLVMAVRInfo;LLVMBPFInfo;LLVMHexagonInfo;LLVMLanaiInfo;LLVMLoongArchInfo;LLVMMipsInfo;LLVMMSP430Info;LLVMNVPTXInfo;LLVMPowerPCInfo;LLVMRISCVInfo;LLVMSparcInfo;LLVMSPIRVInfo;LLVMSystemZInfo;LLVMVEInfo;LLVMWebAssemblyInfo;LLVMX86Info;LLVMXCoreInfo;LLVMFrontendOpenMP;LLVMOption;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclangdRemoteIndex.so.21.0git"
  IMPORTED_SONAME_RELEASE "libclangdRemoteIndex.so.21.0git"
  )

list(APPEND _cmake_import_check_targets clangdRemoteIndex )
list(APPEND _cmake_import_check_files_for_clangdRemoteIndex "${_IMPORT_PREFIX}/lib/libclangdRemoteIndex.so.21.0git" )

# Import target "libclang" for configuration "Release"
set_property(TARGET libclang APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(libclang PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "clangAST;clangBasic;clangDriver;clangExtractAPI;clangFrontend;clangIndex;clangLex;clangRewrite;clangSema;clangSerialization;clangTooling;LLVMAArch64CodeGen;LLVMAArch64AsmParser;LLVMAArch64Desc;LLVMAArch64Disassembler;LLVMAArch64Info;LLVMAArch64Utils;LLVMAMDGPUCodeGen;LLVMAMDGPUAsmParser;LLVMAMDGPUDesc;LLVMAMDGPUDisassembler;LLVMAMDGPUInfo;LLVMAMDGPUUtils;LLVMARMCodeGen;LLVMARMAsmParser;LLVMARMDesc;LLVMARMDisassembler;LLVMARMInfo;LLVMARMUtils;LLVMAVRCodeGen;LLVMAVRAsmParser;LLVMAVRDesc;LLVMAVRDisassembler;LLVMAVRInfo;LLVMBPFCodeGen;LLVMBPFAsmParser;LLVMBPFDesc;LLVMBPFDisassembler;LLVMBPFInfo;LLVMHexagonCodeGen;LLVMHexagonAsmParser;LLVMHexagonDesc;LLVMHexagonDisassembler;LLVMHexagonInfo;LLVMLanaiCodeGen;LLVMLanaiAsmParser;LLVMLanaiDesc;LLVMLanaiDisassembler;LLVMLanaiInfo;LLVMLoongArchCodeGen;LLVMLoongArchAsmParser;LLVMLoongArchDesc;LLVMLoongArchDisassembler;LLVMLoongArchInfo;LLVMMipsCodeGen;LLVMMipsAsmParser;LLVMMipsDesc;LLVMMipsDisassembler;LLVMMipsInfo;LLVMMSP430CodeGen;LLVMMSP430AsmParser;LLVMMSP430Desc;LLVMMSP430Disassembler;LLVMMSP430Info;LLVMNVPTXCodeGen;LLVMNVPTXDesc;LLVMNVPTXInfo;LLVMPowerPCCodeGen;LLVMPowerPCAsmParser;LLVMPowerPCDesc;LLVMPowerPCDisassembler;LLVMPowerPCInfo;LLVMRISCVCodeGen;LLVMRISCVAsmParser;LLVMRISCVDesc;LLVMRISCVDisassembler;LLVMRISCVInfo;LLVMSparcCodeGen;LLVMSparcAsmParser;LLVMSparcDesc;LLVMSparcDisassembler;LLVMSparcInfo;LLVMSPIRVCodeGen;LLVMSPIRVDesc;LLVMSPIRVInfo;LLVMSystemZCodeGen;LLVMSystemZAsmParser;LLVMSystemZDesc;LLVMSystemZDisassembler;LLVMSystemZInfo;LLVMVECodeGen;LLVMVEAsmParser;LLVMVEDesc;LLVMVEDisassembler;LLVMVEInfo;LLVMWebAssemblyCodeGen;LLVMWebAssemblyAsmParser;LLVMWebAssemblyDesc;LLVMWebAssemblyDisassembler;LLVMWebAssemblyInfo;LLVMWebAssemblyUtils;LLVMX86CodeGen;LLVMX86AsmParser;LLVMX86Desc;LLVMX86Disassembler;LLVMX86Info;LLVMXCoreCodeGen;LLVMXCoreDesc;LLVMXCoreDisassembler;LLVMXCoreInfo;LLVMCore;LLVMSupport;LLVMTargetParser"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libclang.so.21.0.0git"
  IMPORTED_SONAME_RELEASE "libclang.so.21.0git"
  )

list(APPEND _cmake_import_check_targets libclang )
list(APPEND _cmake_import_check_files_for_libclang "${_IMPORT_PREFIX}/lib/libclang.so.21.0.0git" )

# Import target "offload-arch" for configuration "Release"
set_property(TARGET offload-arch APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(offload-arch PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/offload-arch"
  )

list(APPEND _cmake_import_check_targets offload-arch )
list(APPEND _cmake_import_check_files_for_offload-arch "${_IMPORT_PREFIX}/bin/offload-arch" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
