function(_add_capi_test_executable name)
  cmake_parse_arguments(ARG
    ""
    ""
    "LINK_LIBS"
    ${ARGN})
  set(LLVM_LINK_COMPONENTS
    )
  add_llvm_executable(${name}
    PARTIAL_SOURCES_INTENDED
    ${ARG_UNPARSED_ARGUMENTS})
  set_target_properties(${name} PROPERTIES FOLDER "AIIR/Tests")
  set_target_properties(${name} PROPERTIES EXCLUDE_FROM_ALL ON)

  llvm_update_compile_flags(${name})
  if(AIIR_BUILD_AIIR_C_DYLIB)
    target_link_libraries(${name} PRIVATE
      LLVMCore
      AIIR-C)
  else()
    target_link_libraries(${name} PRIVATE
      ${ARG_LINK_LIBS})
  endif()
endfunction(_add_capi_test_executable)

if(AIIR_ENABLE_EXECUTION_ENGINE)
  _add_capi_test_executable(aiir-capi-execution-engine-test
    execution_engine.c
  LINK_LIBS PRIVATE
    AIIRCAPIConversion
    AIIRCAPIExecutionEngine
    AIIRCAPIRegisterEverything
)
  _add_capi_test_executable(aiir-capi-global-constructors-test
    global_constructors.c
  LINK_LIBS PRIVATE
    AIIRCAPIConversion
    AIIRCAPIExecutionEngine
    AIIRCAPIRegisterEverything
)
endif()

_add_capi_test_executable(aiir-capi-ir-test
  ir.c
  LINK_LIBS PRIVATE
    AIIRCAPIIR
    AIIRCAPIFunc
    AIIRCAPIRegisterEverything
)

_add_capi_test_executable(aiir-capi-irdl-test
  irdl.c
  LINK_LIBS PRIVATE
    AIIRCAPIIR
    AIIRCAPIIRDL
)

_add_capi_test_executable(aiir-capi-llvm-test
  llvm.c
  LINK_LIBS PRIVATE
    AIIRCAPIIR
    AIIRCAPILLVM
    AIIRCAPIRegisterEverything
)

_add_capi_test_executable(aiir-capi-pass-test
  pass.c
  LINK_LIBS PRIVATE
    AIIRCAPIFunc
    AIIRCAPIIR
    AIIRCAPIRegisterEverything
    AIIRCAPITransforms
)

_add_capi_test_executable(aiir-capi-pdl-test
  pdl.c
  LINK_LIBS PRIVATE
    AIIRCAPIIR
    AIIRCAPIRegisterEverything
    AIIRCAPIPDL
)

_add_capi_test_executable(aiir-capi-sparse-tensor-test
  sparse_tensor.c
  LINK_LIBS PRIVATE
    AIIRCAPIIR
    AIIRCAPIRegisterEverything
    AIIRCAPISparseTensor
)

_add_capi_test_executable(aiir-capi-quant-test
  quant.c
  LINK_LIBS PRIVATE
    AIIRCAPIIR
    AIIRCAPIRegisterEverything
    AIIRCAPIQuant
)

_add_capi_test_executable(aiir-capi-rewrite-test
  rewrite.c
  LINK_LIBS PRIVATE
    AIIRCAPIIR
    AIIRCAPIRegisterEverything
    AIIRCAPITransforms
)


_add_capi_test_executable(aiir-capi-transform-test
  transform.c
  LINK_LIBS PRIVATE
    AIIRCAPIIR
    AIIRCAPIRegisterEverything
    AIIRCAPITransformDialect
)

_add_capi_test_executable(aiir-capi-transform-interpreter-test
  transform_interpreter.c
  LINK_LIBS PRIVATE
    AIIRCAPIIR
    AIIRCAPIRegisterEverything
    AIIRCAPITransformDialect
    AIIRCAPITransformDialectTransforms
)

_add_capi_test_executable(aiir-capi-translation-test
  translation.c
  LINK_LIBS PRIVATE
    AIIRCAPIIR
    AIIRCAPILLVM
    AIIRCAPIRegisterEverything
    AIIRCAPITarget
)

_add_capi_test_executable(aiir-capi-smt-test
  smt.c

  LINK_LIBS PRIVATE
    AIIRCAPIIR
    AIIRCAPIFunc
    AIIRCAPISMT
    AIIRCAPIExportSMTLIB
)
