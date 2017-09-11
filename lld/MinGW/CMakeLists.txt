set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(ShimOptionsTableGen)

if(NOT LLD_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()

add_lld_library(lldMinGW
  Driver.cpp

  LINK_LIBS
  lldConfig
  lldCore
  ${LLVM_PTHREAD_LIB}

  DEPENDS
  ShimOptionsTableGen
  ${tablegen_deps}
)
