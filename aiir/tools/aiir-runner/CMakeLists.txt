set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
  )

add_aiir_tool(aiir-runner
  aiir-runner.cpp

  EXPORT_SYMBOLS
  )
llvm_update_compile_flags(aiir-runner)
aiir_target_link_libraries(aiir-runner PRIVATE
  AIIRAnalysis
  AIIRBuiltinToLLVMIRTranslation
  AIIRIR
  AIIRLLVMDialect
  AIIRLLVMToLLVMIRTranslation
  AIIRToLLVMIRTranslationRegistration
  AIIRParser
  AIIRTargetLLVMIRExport
  AIIRSupport
  )
target_link_libraries(aiir-runner PRIVATE
  AIIRExecutionEngine
  AIIRJitRunner
  )

# Preventing re-export of symbols causes link errors with ASan and UBSan libs.
if (NOT LLVM_USE_SANITIZER)
  target_link_options(aiir-runner
    PRIVATE
      # On Linux, disable re-export of any static linked libraries that came
      # through. This prevents our LLVM build from interfering with the version
      # of LLVM included in certain graphics drivers.
      $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>
  )
endif()
