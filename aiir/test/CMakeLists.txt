include(AIIRCheckHardwareFeatures)

add_subdirectory(CAPI)
add_subdirectory(lib)

if (AIIR_ENABLE_BINDINGS_PYTHON)
  add_subdirectory(python)
endif()

# Provide the AIIR CMake module dir so that the out of tree Standalone
# dialect and can add it to the module path.
set(AIIR_CMAKE_DIR
  "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/aiir")

# Passed to lit.site.cfg.py.in to set up the path where to find libraries.
set(AIIR_LIB_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

if (AIIR_INCLUDE_INTEGRATION_TESTS)
  set(INTEL_SDE_EXECUTABLE "" CACHE STRING
      "If set, arch-specific integration tests are run with Intel SDE.")
  set(ARM_EMULATOR_EXECUTABLE "" CACHE STRING
      "If set, arch-specific Arm integration tests are run with an emulator.")
  set(ARM_EMULATOR_OPTIONS "" CACHE STRING
      "If arch-specific Arm integration tests run emulated, pass these as parameters to the emulator.")
  set(ARM_EMULATOR_AIIR_RUNNER_EXECUTABLE "" CACHE STRING
      "If arch-specific Arm integration tests run emulated, use this Arm native aiir-runner.")
  set(ARM_EMULATOR_LLI_EXECUTABLE "" CACHE STRING
      "If arch-specific Arm integration tests run emulated, use this Arm native lli.")
  set(ARM_EMULATOR_UTILS_LIB_DIR "" CACHE STRING
      "If arch-specific Arm integration tests run emulated, find Arm native utility libraries in this directory.")
  set(AIIR_GPU_COMPILATION_TEST_FORMAT "fatbin" CACHE STRING
      "The GPU compilation format used by the tests.")
  set(ARM_SME_ABI_ROUTINES_SHLIB "" CACHE STRING
      "Path to a shared library containing Arm SME ABI routines, required for Arm SME integration tests.")
  option(AIIR_RUN_X86_TESTS "Run X86 tests.")
  option(AIIR_RUN_X86_AMX_TESTS "Run X86 AMX tests.")
  option(AIIR_RUN_CUDA_TENSOR_CORE_TESTS "Run CUDA Tensor core WMMA tests.")
  option(AIIR_RUN_CUDA_SM80_TESTS "Run CUDA A100 tests.")
  option(AIIR_RUN_CUDA_SM80_LT_TESTS "Run CUDA A100 structured sparsity tests.")
  option(AIIR_RUN_CUDA_SM90_TESTS "Run CUDA H100 tests.")
  option(AIIR_RUN_ARM_SVE_TESTS "Run Arm SVE tests.")
  option(AIIR_RUN_ARM_SME_TESTS "Run Arm SME tests.")

  # Check whether an emulator is required - if yes then make sure that it's
  # been set.
  check_emulator(AIIR_RUN_ARM_SVE_TESTS "HWCAP_SVE" ARM_EMULATOR_EXECUTABLE)
  check_emulator(AIIR_RUN_ARM_SME_TESTS "HWCAP2_SME" ARM_EMULATOR_EXECUTABLE)

  # The native target may not be enabled when cross compiling, raise an error.
  if(NOT AIIR_ENABLE_EXECUTION_ENGINE)
    message(FATAL_ERROR "AIIR_INCLUDE_INTEGRATION_TESTS requires a native target")
  endif()

  # When the Integration tests are requested via the AIIR_INCLUDE_INTEGRATION_TESTS
  # configuration flag, we automatically include sm80 tests when build for
  # cuSparse when the configuration flag AIIR_ENABLE_CUDA_CUSPARSE is set and
  # include sm80 lt tests when the AIIR_ENABLE_CUDA_CUSPARSELT is set in
  # addition to those.
  if (AIIR_ENABLE_CUDA_CUSPARSE)
    set(AIIR_RUN_CUDA_SM80_TESTS ON)
    if (AIIR_ENABLE_CUDA_CUSPARSELT)
      set(AIIR_RUN_CUDA_SM80_LT_TESTS ON)
    endif()
  endif()

endif()

llvm_canonicalize_cmake_booleans(
  LLVM_BUILD_EXAMPLES
  LLVM_HAS_NVPTX_TARGET
  LLVM_INCLUDE_SPIRV_TOOLS_TESTS
  LLVM_HAS_AMDGPU_TARGET
  AIIR_ENABLE_BINDINGS_PYTHON
  AIIR_ENABLE_PYTHON_STABLE_ABI
  AIIR_ENABLE_CUDA_RUNNER
  AIIR_ENABLE_ROCM_RUNNER
  AIIR_ENABLE_SPIRV_CPU_RUNNER
  AIIR_ENABLE_VULKAN_RUNNER
  AIIR_INCLUDE_INTEGRATION_TESTS
  AIIR_RUN_X86_AMX_TESTS
  AIIR_RUN_CUDA_TENSOR_CORE_TESTS
  AIIR_RUN_X86_TESTS
  AIIR_RUN_ARM_SVE_TESTS
  AIIR_RUN_ARM_SME_TESTS
  AIIR_RUN_CUDA_SM80_TESTS
  AIIR_RUN_CUDA_SM80_LT_TESTS
  AIIR_RUN_CUDA_SM90_TESTS
  BUILD_SHARED_LIBS
  )

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
  )

set(AIIR_TEST_DEPENDS
  aiir-capi-ir-test
  aiir-capi-irdl-test
  aiir-capi-llvm-test
  aiir-capi-pass-test
  aiir-capi-quant-test
  aiir-capi-rewrite-test
  aiir-capi-smt-test
  aiir-capi-sparse-tensor-test
  aiir-capi-transform-test
  aiir-capi-transform-interpreter-test
  aiir-capi-translation-test
  aiir-linalg-ods-yaml-gen
  aiir-lsp-server
  aiir-opt
  aiir-query
  aiir-reduce
  aiir-rewrite
  aiir-tblgen
  aiir-translate
  tblgen-lsp-server
  tblgen-to-irdl
  )
if(NOT AIIR_STANDALONE_BUILD)
  list(APPEND AIIR_TEST_DEPENDS
    FileCheck
    count
    llvm-strings
    not
    split-file
    yaml2obj
    )
endif()
# Examples/standalone/test.toy (vis-a-vis the standalone example) depends on these.
if(LLVM_INCLUDE_EXAMPLES)
  list(APPEND AIIR_TEST_DEPENDS AIIRCAPIArith)
endif()

set(AIIR_TEST_DEPENDS ${AIIR_TEST_DEPENDS}
  aiir-capi-pdl-test
  aiir-pdll-lsp-server
  aiir-pdll
  )

# The native target may not be enabled, in this case we won't
# run tests that involves executing on the host: do not build
# useless binaries.
if(LLVM_ENABLE_PIC AND TARGET ${LLVM_NATIVE_ARCH})
  list(APPEND AIIR_TEST_DEPENDS
    aiir-runner
    llc
    aiir_async_runtime
    aiir-capi-execution-engine-test
    aiir-capi-global-constructors-test
    aiir_c_runner_utils
    aiir_runner_utils
    aiir_float16_utils
  )
endif()

if (AIIR_INCLUDE_INTEGRATION_TESTS)
  list(APPEND AIIR_TEST_DEPENDS lli)
endif()

if(AIIR_ENABLE_CUDA_RUNNER)
  list(APPEND AIIR_TEST_DEPENDS aiir_cuda_runtime)
endif()

if(AIIR_ENABLE_EXECUTION_ENGINE)
  list(APPEND AIIR_TEST_DEPENDS
    aiir-capi-execution-engine-test
    aiir-capi-global-constructors-test
  )
endif()

if(AIIR_ENABLE_ROCM_RUNNER)
  list(APPEND AIIR_TEST_DEPENDS aiir_rocm_runtime)
endif()

if(AIIR_ENABLE_SYCL_RUNNER)
  list(APPEND AIIR_TEST_DEPENDS aiir_sycl_runtime)
endif()

if(AIIR_ENABLE_LEVELZERO_RUNNER)
  list(APPEND AIIR_TEST_DEPENDS aiir_levelzero_runtime)
endif()

if (AIIR_RUN_ARM_SME_TESTS AND NOT ARM_SME_ABI_ROUTINES_SHLIB)
  list(APPEND AIIR_TEST_DEPENDS aiir_arm_sme_abi_stubs)
endif()

if (AIIR_RUN_ARM_SVE_TESTS OR AIIR_RUN_ARM_SME_TESTS)
  list(APPEND AIIR_TEST_DEPENDS aiir_arm_runner_utils)
endif()

list(APPEND AIIR_TEST_DEPENDS AIIRUnitTests)

if(LLVM_BUILD_EXAMPLES)
  list(APPEND AIIR_TEST_DEPENDS
    toyc-ch1
    toyc-ch2
    toyc-ch3
    toyc-ch4
    toyc-ch5
    )
  list(APPEND AIIR_TEST_DEPENDS
    transform-opt-ch2
    transform-opt-ch3
    transform-opt-ch4
    aiir-minimal-opt
    aiir-transform-opt
    )
  if(AIIR_ENABLE_EXECUTION_ENGINE)
    list(APPEND AIIR_TEST_DEPENDS
      toyc-ch6
      toyc-ch7
    )
  endif()
endif()

if(AIIR_ENABLE_SPIRV_CPU_RUNNER)
  list(APPEND AIIR_TEST_DEPENDS
    aiir_spirv_cpu_runtime
  )
endif()

if(AIIR_ENABLE_VULKAN_RUNNER)
  list(APPEND AIIR_TEST_DEPENDS
    aiir_vulkan_runtime
  )
endif()

if(AIIR_ENABLE_BINDINGS_PYTHON)
  list(APPEND AIIR_TEST_DEPENDS
    AIIRPythonModules
  )
endif()

if (LLVM_INCLUDE_SPIRV_TOOLS_TESTS)
  list(APPEND AIIR_TEST_DEPENDS spirv-as)
  list(APPEND AIIR_TEST_DEPENDS spirv-val)
endif()

# This target can be used to just build the dependencies
# for the check-aiir target without executing the tests.
# This is useful for bots when splitting the build step
# from the testing step.
add_custom_target(check-aiir-build-only
  DEPENDS ${AIIR_TEST_DEPENDS}
)
set_target_properties(check-aiir-build-only PROPERTIES FOLDER "AIIR/Tests")

add_lit_testsuite(check-aiir "Running the AIIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${AIIR_TEST_DEPENDS}
  )

add_lit_testsuites(AIIR ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${AIIR_TEST_DEPENDS}
  SKIP "^python*"
)
