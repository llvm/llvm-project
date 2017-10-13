if(NOT LLD_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()

add_lld_library(lldCommon
  Reproduce.cpp
  TargetOptionsCommandFlags.cpp
  Threads.cpp
  Version.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLD_INCLUDE_DIR}/lld/Common

  LINK_COMPONENTS
  Codegen
  MC
  Option
  Support
  Target

  DEPENDS
  ${tablegen_deps}
  )
