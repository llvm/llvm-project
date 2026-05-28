add_subdirectory(IR)
add_subdirectory(OpenACC)
add_subdirectory(OpenMP)
add_subdirectory(Transforms)

add_clang_library(CIRDialectRegistration
  CIRDialectRegistration.cpp

  LINK_LIBS PUBLIC
  MLIRCIR
  CIROpenACCSupport
  CIROpenMPSupport
  MLIRDLTIDialect
)
