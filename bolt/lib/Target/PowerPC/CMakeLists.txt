add_llvm_library(LLVMBOLTTargetPowerPC
    PPCMCPlusBuilder.cpp
)

target_include_directories(LLVMBOLTTargetPowerPC PRIVATE
    ${LLVM_BINARY_DIR}/include
    ${LLVM_SOURCE_DIR}/include
)

file(MAKE_DIRECTORY "${LLVM_BINARY_DIR}/include/llvm/Target/PowerPC")

foreach(incfile IN ITEMS
    PPCGenInstrInfo.inc
    PPCGenRegisterInfo.inc
)
    add_custom_command(
        OUTPUT "${LLVM_BINARY_DIR}/include/llvm/Target/PowerPC/${incfile}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${LLVM_BINARY_DIR}/lib/Target/PowerPC/${incfile}"
            "${LLVM_BINARY_DIR}/include/llvm/Target/PowerPC/${incfile}"
        DEPENDS "${LLVM_BINARY_DIR}/lib/Target/PowerPC/${incfile}"
        COMMENT "Copying ${incfile} to include directory"
    )
    add_custom_target(
        "BoltCopy${incfile}" ALL
        DEPENDS "${LLVM_BINARY_DIR}/include/llvm/Target/PowerPC/${incfile}"
    )
    add_dependencies(LLVMBOLTTargetPowerPC "BoltCopy${incfile}")
endforeach()