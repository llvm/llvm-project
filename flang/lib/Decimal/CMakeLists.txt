
add_library(FortranDecimal
  binary-to-decimal.cpp
  decimal-to-binary.cpp
)

target_compile_features(FortranDecimal PUBLIC cxx_std_17)

install (TARGETS FortranDecimal
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
