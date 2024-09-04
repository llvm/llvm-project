add_flang_library(FortranSupport
  Timing.cpp

  LINK_LIBS
  MLIRSupport

  LINK_COMPONENTS
  Support
)
