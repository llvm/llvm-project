add_flang_library(FIRTestOpenACCInterfaces
  TestOpenACCInterfaces.cpp

  DEPENDS
  FIRDialect
  FIRBuilder
  FIROpenACCSupport
  FIRSupport
  FIRTransforms
  MLIROpenACCDialect

  LINK_LIBS
  FIRDialect
  FIRBuilder
  FIROpenACCSupport
  FIRSupport
  MLIRIR
  MLIROpenACCDialect
  MLIRSupport
  MLIRFuncDialect
)
