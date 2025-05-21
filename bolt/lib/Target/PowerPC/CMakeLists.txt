add_llvm_library(LLVMBOLTTargetPowerPC
    PPCMCPlusBuilder.cpp
)

target_include_directories(LLVMBOLTTargetPowerPC PRIVATE
    ${LLVM_BINARY_DIR}/include
    ${LLVM_SOURCE_DIR}/include
)

# Ensure the destination directory exists
file(MAKE_DIRECTORY "${LLVM_BINARY_DIR}/include/llvm/Target/PowerPC")

# Custom command to copy the .inc file when it's generated
add_custom_command(
    OUTPUT "${LLVM_BINARY_DIR}/include/llvm/Target/PowerPC/PPCGenInstrInfo.inc"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LLVM_BINARY_DIR}/lib/Target/PowerPC/PPCGenInstrInfo.inc"
        "${LLVM_BINARY_DIR}/include/llvm/Target/PowerPC/PPCGenInstrInfo.inc"
    DEPENDS "${LLVM_BINARY_DIR}/lib/Target/PowerPC/PPCGenInstrInfo.inc"
    COMMENT "Copying PPCGenInstrInfo.inc to include directory"
)

# Custom target to represent the copied .inc file
add_custom_target(
    BoltCopyPPCGenInstrInfoInc ALL
    DEPENDS "${LLVM_BINARY_DIR}/include/llvm/Target/PowerPC/PPCGenInstrInfo.inc"
)

# Make your library depend on the copy
add_dependencies(LLVMBOLTTargetPowerPC BoltCopyPPCGenInstrInfoInc)