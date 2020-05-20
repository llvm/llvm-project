
add_flang_library(FortranCommon
  Fortran.cpp
  Fortran-features.cpp
  default-kinds.cpp
  idioms.cpp

  LINK_COMPONENTS
  Support
)
