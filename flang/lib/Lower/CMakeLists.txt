set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -Wno-unused-parameter")

add_flang_library(FortranLower
  ConvertType.cpp
  ComplexExpr.cpp
  OpenMP.cpp
  PFTBuilder.cpp

  LINK_COMPONENTS
  Support
)
