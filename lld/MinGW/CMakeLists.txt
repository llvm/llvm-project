set(LLVM_TARGET_DEFINITIONS Options.td)
llvm_tablegen(Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(MinGWOptionsTableGen)

add_lld_library(lldMinGW
  Driver.cpp

  LINK_COMPONENTS
  Option
  Support
  TargetParser

  LINK_LIBS
  lldCOFF
  lldCommon

  DEPENDS
  MinGWOptionsTableGen
  intrinsics_gen
)
