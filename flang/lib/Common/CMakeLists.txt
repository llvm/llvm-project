
add_library(FortranCommon
  Fortran.cpp
  Fortran-features.cpp
  default-kinds.cpp
  idioms.cpp
)

target_compile_features(FortranCommon PUBLIC cxx_std_17)

install (TARGETS FortranCommon
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
