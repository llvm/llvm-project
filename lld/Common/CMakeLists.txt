add_lld_library(lldCommon
  Reproduce.cpp
  TargetOptionsCommandFlags.cpp
  Version.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLD_INCLUDE_DIR}/lld/Common

  LINK_COMPONENTS
  Codegen
  MC
  Option
  Support
  Target
  )
