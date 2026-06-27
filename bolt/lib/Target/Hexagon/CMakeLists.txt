set(LLVM_LINK_COMPONENTS
  MC
  Support
  HexagonDesc
  )

if(BOLT_BUILT_STANDALONE)
  # tablegen, copied from llvm/lib/Target/Hexagon/CMakeLists.txt
  set(LLVM_TARGET_DEFINITIONS ${LLVM_MAIN_SRC_DIR}/lib/Target/Hexagon/Hexagon.td)
  list(APPEND LLVM_TABLEGEN_FLAGS -I ${LLVM_MAIN_SRC_DIR}/lib/Target/Hexagon)
  tablegen(LLVM HexagonGenInstrInfo.inc -gen-instr-info)
  tablegen(LLVM HexagonGenRegisterInfo.inc -gen-register-info)
  tablegen(LLVM HexagonGenSubtargetInfo.inc -gen-subtarget)

  add_public_tablegen_target(HexagonCommonTableGen)
  include_directories(${CMAKE_CURRENT_BINARY_DIR})
endif()

add_llvm_library(LLVMBOLTTargetHexagon
  HexagonMCPlusBuilder.cpp

  NO_EXPORT
  DISABLE_LLVM_LINK_LLVM_DYLIB

  DEPENDS
  HexagonCommonTableGen
  )

target_link_libraries(LLVMBOLTTargetHexagon PRIVATE LLVMBOLTCore)

include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/Hexagon
  ${LLVM_BINARY_DIR}/lib/Target/Hexagon
  )
