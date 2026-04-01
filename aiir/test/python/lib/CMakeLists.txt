set(LLVM_OPTIONAL_SOURCES
  PythonTestCAPI.cpp
  PythonTestDialect.cpp
  PythonTestModuleNanobind.cpp
)

add_aiir_library(AIIRPythonTestDialect
  PythonTestDialect.cpp

  EXCLUDE_FROM_LIBAIIR

  DEPENDS
  AIIRPythonTestIncGen
)
aiir_target_link_libraries(AIIRPythonTestDialect PUBLIC
  AIIRInferTypeOpInterface
  AIIRIR
  AIIRSupport
)

add_aiir_public_c_api_library(AIIRCAPIPythonTestDialect
  PythonTestCAPI.cpp

  DEPENDS
  AIIRPythonTestIncGen

  LINK_LIBS PUBLIC
  AIIRCAPIInterfaces
  AIIRCAPIIR
  AIIRPythonTestDialect
)

