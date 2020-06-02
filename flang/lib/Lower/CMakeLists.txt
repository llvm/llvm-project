
add_flang_library(FortranLower
  OpenMP.cpp
  PFTBuilder.cpp

  LINK_COMPONENTS
  Support
)
