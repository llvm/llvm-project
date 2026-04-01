get_property(conversion_libs GLOBAL PROPERTY AIIR_CONVERSION_LIBS)
add_aiir_upstream_c_api_library(AIIRCAPIConversion
  Passes.cpp

  DEPENDS
  AIIRConversionPassIncGen

  LINK_LIBS PUBLIC
  ${conversion_libs}
)
