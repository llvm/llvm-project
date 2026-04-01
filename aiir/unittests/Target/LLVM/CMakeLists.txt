set(LLVM_LINK_COMPONENTS nativecodegen BitReader)

add_aiir_unittest(AIIRTargetLLVMTests
  SerializeNVVMTarget.cpp
  SerializeROCDLTarget.cpp
  SerializeToLLVMBitcode.cpp
DEPENDS
  AIIRRegisterAllDialects
)

aiir_target_link_libraries(AIIRTargetLLVMTests
  PRIVATE
  AIIRTargetLLVM
  AIIRNVVMTarget
  AIIRROCDLTarget
  AIIRGPUDialect
  AIIRNVVMDialect
  AIIRLLVMDialect
  AIIRLLVMToLLVMIRTranslation
  AIIRBuiltinToLLVMIRTranslation
  AIIRNVVMToLLVMIRTranslation
  AIIRROCDLToLLVMIRTranslation
  AIIRGPUToLLVMIRTranslation
)

if (DEFINED LLVM_NATIVE_TARGET)
  target_compile_definitions(AIIRTargetLLVMTests
    PRIVATE
    -DLLVM_NATIVE_TARGET_TEST_ENABLED=1
  )
else()
  target_compile_definitions(AIIRTargetLLVMTests
    PRIVATE
    -DLLVM_NATIVE_TARGET_TEST_ENABLED=0
  )
endif()
