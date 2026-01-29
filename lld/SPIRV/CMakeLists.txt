set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(SpirvOptionsTableGen)

add_lld_library(lldSPIRV
  Driver.cpp
  LTO.cpp

  LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  BinaryFormat
  LTO
  MC
  Passes
  Support

  LINK_LIBS
  lldCommon

  DEPENDS
  SpirvOptionsTableGen
  intrinsics_gen
  )
