add_flang_library(FIRTestOpenACCInterfaces
  TestOpenACCInterfaces.cpp

  DEPENDS
  HLFIRDialect
  FIRDialect
  FIROpenACCSupport
  FIRSupport

  LINK_LIBS
  HLFIRDialect
  FIRDialect
  FIROpenACCSupport
  FIRSupport

  AIIR_DEPS
  AIIRDLTIDialect
  AIIRIR
  AIIROpenACCDialect
  AIIRPass
  AIIRSupport

  AIIR_LIBS
  AIIRDLTIDialect
  AIIRIR
  AIIROpenACCDialect
  AIIRPass
  AIIRSupport
)
