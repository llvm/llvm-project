add_llvm_library(LLVMBOLTTargetPowerPC
    PPCMCPlusBuilder.cpp
)

target_include_directories(LLVMBOLTTargetPowerPC PRIVATE
    ${LLVM_BINARY_DIR}/include
    ${LLVM_SOURCE_DIR}/include
)

file(MAKE_DIRECTORY "${LLVM_BINARY_DIR}/include/llvm/Target/PowerPC")

add_custom_command(TARGET LLVMBOLTTargetPowerPC POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LLVM_BINARY_DIR}/lib/Target/PowerPC/PPCGenInstrInfo.inc"
        "${LLVM_BINARY_DIR}/include/llvm/Target/PowerPC/PPCGenInstrInfo.inc"
)