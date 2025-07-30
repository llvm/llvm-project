set(BOLTTargetPowerPCTestsSources
    PPCMCPlusBuilderTest.cpp)

add_bolt_unittest(BOLTTargetPowerPCTests
  ${BOLTTargetPowerPCTestsSources}
)

target_link_libraries(BOLTTargetPowerPCTests PRIVATE
  LLVMBOLTTargetPowerPC
  LLVMBOLTCore
  LLVMCore
)

target_include_directories(BOLTTargetPowerPCTests PRIVATE
  ${LLVM_BINARY_DIR}/include
  ${LLVM_SOURCE_DIR}/include
  ${LLVM_SOURCE_DIR}/bolt/include
  ${LLVM_BINARY_DIR}/tools/bolt/include
  ${CMAKE_SOURCE_DIR}
)
