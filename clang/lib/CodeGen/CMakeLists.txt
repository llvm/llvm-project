set(LLVM_LINK_COMPONENTS
  Analysis
  BitReader
  BitWriter
  Core
  Coroutines
  Coverage
  IPO
  IRReader
  InstCombine
  Instrumentation
  LTO
  Linker
  MC
  ObjCARCOpts
  Object
  Passes
  ProfileData
  ScalarOpts
  Support
  Target
  TransformUtils
  )

# In a standard Clang+LLVM build, we need to generate intrinsics before
# building codegen. In a standalone build, LLVM is already built and we don't
# need this dependency. Furthermore, LLVM doesn't export it so we can't have
# this dependency.
set(codegen_deps intrinsics_gen)
if (CLANG_BUILT_STANDALONE)
  set(codegen_deps)
endif()

add_clang_library(clangCodeGen
  BackendUtil.cpp
  CGAtomic.cpp
  CGBlocks.cpp
  CGBuiltin.cpp
  CGCUDANV.cpp
  CGCUDARuntime.cpp
  CGCXX.cpp
  CGCXXABI.cpp
  CGCall.cpp
  CGClass.cpp
  CGCleanup.cpp
  CGCoroutine.cpp
  CGDebugInfo.cpp
  CGDecl.cpp
  CGDeclCXX.cpp
  CGException.cpp
  CGExpr.cpp
  CGExprAgg.cpp
  CGExprCXX.cpp
  CGExprComplex.cpp
  CGExprConstant.cpp
  CGExprScalar.cpp
  CGGPUBuiltin.cpp
  CGLoopInfo.cpp
  CGNonTrivialStruct.cpp
  CGObjC.cpp
  CGObjCGNU.cpp
  CGObjCMac.cpp
  CGObjCRuntime.cpp
  CGOpenCLRuntime.cpp
  CGOpenMPRuntime.cpp
  CGOpenMPRuntimeNVPTX.cpp
  CGRecordLayoutBuilder.cpp
  CGStmt.cpp
  CGStmtOpenMP.cpp
  CGVTT.cpp
  CGVTables.cpp
  CodeGenABITypes.cpp
  CodeGenAction.cpp
  CodeGenFunction.cpp
  CodeGenModule.cpp
  CodeGenPGO.cpp
  CodeGenTBAA.cpp
  CodeGenTypes.cpp
  ConstantInitBuilder.cpp
  CoverageMappingGen.cpp
  ItaniumCXXABI.cpp
  MacroPPCallbacks.cpp
  MicrosoftCXXABI.cpp
  ModuleBuilder.cpp
  ObjectFilePCHContainerOperations.cpp
  SanitizerMetadata.cpp
  SwiftCallingConv.cpp
  TargetInfo.cpp
  VarBypassDetector.cpp

  DEPENDS
  ${codegen_deps}

  LINK_LIBS
  clangAnalysis
  clangAST
  clangBasic
  clangFrontend
  clangLex
  )
