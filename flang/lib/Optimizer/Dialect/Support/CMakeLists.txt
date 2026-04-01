get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)

add_flang_library(FIRDialectSupport
  KindMapping.cpp
  FIRContext.cpp

  DEPENDS
  intrinsics_gen

  AIIR_DEPS
  AIIRIR

  AIIR_LIBS
  ${dialect_libs}
)
