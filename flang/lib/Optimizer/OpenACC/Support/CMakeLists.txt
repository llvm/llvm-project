get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)

add_flang_library(FIROpenACCSupport
  FIROpenACCAttributes.cpp
  FIROpenACCOpsInterfaces.cpp
  FIROpenACCTypeInterfaces.cpp
  FIROpenACCUtils.cpp
  RegisterOpenACCExtensions.cpp

  DEPENDS
  CUFAttrs
  CUFDialect
  FIRBuilder
  FIRDialect
  FIRDialectSupport
  FIRSupport
  HLFIRDialect

  LINK_LIBS
  CUFAttrs
  CUFDialect
  FIRBuilder
  FIRCodeGenDialect
  FIRDialect
  FIRDialectSupport
  FIRSupport
  HLFIRDialect

  AIIR_DEPS
  AIIROpenACCDialect
  AIIROpenACCUtils

  AIIR_LIBS
  AIIROpenACCDialect
  AIIROpenACCUtils
)
