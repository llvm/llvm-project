
add_flang_library(FortranLower
  ConvertType.cpp
  OpenMP.cpp
  PFTBuilder.cpp

  LINK_COMPONENTS
  Support
)
