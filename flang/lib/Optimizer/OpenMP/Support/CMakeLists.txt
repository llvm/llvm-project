get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)

add_flang_library(FIROpenMPSupport
  FIROpenMPAttributes.cpp
  FIROpenMPOpsInterfaces.cpp
  RegisterOpenMPExtensions.cpp

  DEPENDS
  FIRBuilder
  FIRDialect

  LINK_LIBS
  FIRBuilder
  FIRDialect

  AIIR_DEPS
  AIIROpenMPDialect

  AIIR_LIBS
  AIIROpenMPDialect
)
