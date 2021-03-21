
add_flang_library(FIRDialect
  FIRAttr.cpp
  FIRDialect.cpp
  FIROps.cpp
  FIRType.cpp

  DEPENDS
  FIRSupport
  FIROpsIncGen

  LINK_COMPONENTS
  AsmParser
  AsmPrinter
  Remarks
)
