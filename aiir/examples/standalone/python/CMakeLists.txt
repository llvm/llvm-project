include(AddAIIRPython)

# Specifies that all AIIR packages are co-located under the `aiir_standalone`
# top level package (the API has been embedded in a relocatable way).
add_compile_definitions("AIIR_PYTHON_PACKAGE_PREFIX=${AIIR_PYTHON_PACKAGE_PREFIX}.")


################################################################################
# Sources
################################################################################

declare_aiir_python_sources(StandalonePythonSources)

declare_aiir_dialect_python_bindings(
  ADD_TO_PARENT StandalonePythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/aiir_standalone"
  TD_FILE dialects/StandaloneOps.td
  SOURCES
    dialects/standalone_nanobind.py
    _aiir_libs/_standaloneDialectsNanobind/py.typed
  DIALECT_NAME standalone)

declare_aiir_python_extension(StandalonePythonSources.NanobindExtension
  MODULE_NAME _standaloneDialectsNanobind
  ADD_TO_PARENT StandalonePythonSources
  SOURCES
    StandaloneExtensionNanobind.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    AIIRCAPIIR
    AIIRCAPIArith
    AIIRCAPITransforms
    StandaloneCAPI
)


################################################################################
# Common CAPI
################################################################################

add_aiir_python_common_capi_library(StandalonePythonCAPI
  INSTALL_COMPONENT StandalonePythonModules
  INSTALL_DESTINATION "${AIIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_aiir_libs"
  OUTPUT_DIRECTORY "${AIIR_BINARY_DIR}/${AIIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_aiir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    StandalonePythonSources
    AIIRPythonSources.Core
    AIIRPythonSources.Dialects.builtin
)

################################################################################
# Instantiation of all Python modules
################################################################################

set(StandalonePythonModules_ROOT_PREFIX "${AIIR_BINARY_DIR}/${AIIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

if(AIIR_PYTHON_STUBGEN_ENABLED)
  # Everything here is very tightly coupled. See the ample descriptions at the bottom of
  # aiir/python/CMakeLists.txt.

  # For a non-external projects build (e.g., installed distro) the type gen targets for the core _aiir module
  # need to be re-declared. On the contrary, for an external projects build, the AIIRPythonExtension.Core.type_stub_gen
  # target already exists and can just be added to DECLARED_SOURCES (see below).
  if(NOT EXTERNAL_PROJECT_BUILD)
    set(_core_type_stub_sources
      _aiir/__init__.pyi
      _aiir/ir.pyi
      _aiir/passmanager.pyi
      _aiir/rewrite.pyi
    )
    get_target_property(_core_extension_srcs AIIRPythonExtension.Core INTERFACE_SOURCES)
    aiir_generate_type_stubs(
      MODULE_NAME _aiir
      DEPENDS_TARGETS StandalonePythonModules.extension._aiir.dso
      OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_aiir_libs"
      OUTPUTS "${_core_type_stub_sources}"
      DEPENDS_TARGET_SRC_DEPS "${_core_extension_srcs}"
      IMPORT_PATHS "${StandalonePythonModules_ROOT_PREFIX}/_aiir_libs"
      VERBOSE
    )
    set(_aiir_typestub_gen_target "${NB_STUBGEN_CUSTOM_TARGET}")

    list(TRANSFORM _core_type_stub_sources PREPEND "_aiir_libs/")
    declare_aiir_python_sources(
      StandalonePythonExtension.Core.type_stub_gen
      ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
      ADD_TO_PARENT StandalonePythonSources
      SOURCES "${_core_type_stub_sources}"
    )
  endif()

  get_target_property(_standalone_extension_srcs StandalonePythonSources.NanobindExtension INTERFACE_SOURCES)
  aiir_generate_type_stubs(
    MODULE_NAME aiir_standalone._aiir_libs._standaloneDialectsNanobind
    DEPENDS_TARGETS
      StandalonePythonModules.extension._aiir.dso
      StandalonePythonModules.extension._standaloneDialectsNanobind.dso
    OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs/_aiir_libs"
    OUTPUTS
      _standaloneDialectsNanobind/__init__.pyi
      _standaloneDialectsNanobind/standalone.pyi
    DEPENDS_TARGET_SRC_DEPS "${_standalone_extension_srcs}"
    IMPORT_PATHS "${StandalonePythonModules_ROOT_PREFIX}/.."
  )
  set(_standaloneDialectsNanobind_typestub_gen_target "${NB_STUBGEN_CUSTOM_TARGET}")

  declare_aiir_python_sources(
    StandalonePythonSources.type_stub_gen
    ROOT_DIR "${CMAKE_CURRENT_BINARY_DIR}/type_stubs"
    ADD_TO_PARENT StandalonePythonSources
    SOURCES
      _aiir_libs/_standaloneDialectsNanobind/__init__.pyi
      _aiir_libs/_standaloneDialectsNanobind/standalone.pyi
  )
endif()

set(_declared_sources
  StandalonePythonSources
  AIIRPythonSources.Core
  AIIRPythonSources.Dialects.builtin
)
# For an external projects build, the AIIRPythonExtension.Core.type_stub_gen
# target already exists and can just be added to DECLARED_SOURCES.
if(EXTERNAL_PROJECT_BUILD AND AIIR_PYTHON_STUBGEN_ENABLED)
  list(APPEND _declared_sources AIIRPythonExtension.Core.type_stub_gen)
endif()

add_aiir_python_modules(StandalonePythonModules
  ROOT_PREFIX "${StandalonePythonModules_ROOT_PREFIX}"
  INSTALL_PREFIX "${AIIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES "${_declared_sources}"
  COMMON_CAPI_LINK_LIBS
    StandalonePythonCAPI
)

if(AIIR_PYTHON_STUBGEN_ENABLED)
  if(NOT EXTERNAL_PROJECT_BUILD)
    add_dependencies(StandalonePythonModules "${_aiir_typestub_gen_target}")
  endif()
  add_dependencies(StandalonePythonModules "${_standaloneDialectsNanobind_typestub_gen_target}")
endif()
