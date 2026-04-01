# Exclude these from libAIIR.so because the JIT infrastructure
# is a big dependency which most don't need.

set(LLVM_OPTIONAL_SOURCES
  APFloatWrappers.cpp
  ArmRunnerUtils.cpp
  ArmSMEStubs.cpp
  AsyncRuntime.cpp
  CRunnerUtils.cpp
  CudaRuntimeWrappers.cpp
  SparseTensorRuntime.cpp
  ExecutionEngine.cpp
  Float16bits.cpp
  RocmRuntimeWrappers.cpp
  RunnerUtils.cpp
  OptUtils.cpp
  JitRunner.cpp
  LevelZeroRuntimeWrappers.cpp
  SpirvCpuRuntimeWrappers.cpp
  SyclRuntimeWrappers.cpp
  VulkanRuntimeWrappers.cpp
  VulkanRuntime.cpp
  VulkanRuntime.h
  )

# Use a separate library for OptUtils, to avoid pulling in the entire JIT and
# codegen infrastructure. Unlike AIIRExecutionEngine, this is part of
# libAIIR.so.
add_aiir_library(AIIRExecutionEngineUtils
  OptUtils.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/ExecutionEngine

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  Core
  Coroutines
  AggressiveInstCombine
  InstCombine
  ScalarOpts
  Vectorize
  TransformUtils
  IPO
  Passes
  TargetParser
  )

if(NOT AIIR_ENABLE_EXECUTION_ENGINE)
  return()
endif()

if(LLVM_USE_INTEL_JITEVENTS)
  set(LLVM_JIT_LISTENER_LIB
      IntelJITEvents)
endif(LLVM_USE_INTEL_JITEVENTS)

if(LLVM_USE_PERF)
  set(LLVM_JIT_LISTENER_LIB
      PerfJITEvents)
endif(LLVM_USE_PERF)

add_aiir_library(AIIRExecutionEngine
  ExecutionEngine.cpp

  EXCLUDE_FROM_LIBAIIR

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/ExecutionEngine

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  Coroutines
  ExecutionEngine
  Object
  OrcJIT
  JITLink
  Analysis
  AggressiveInstCombine
  InstCombine
  MC
  ScalarOpts
  Target
  Vectorize
  TransformUtils
  nativecodegen
  IPO
  Passes
  ${LLVM_JIT_LISTENER_LIB}
  )

aiir_target_link_libraries(AIIRExecutionEngine PUBLIC
  AIIRBuiltinToLLVMIRTranslation
  AIIRExecutionEngineUtils
  AIIRLLVMDialect
  AIIRLLVMToLLVMIRTranslation
  AIIROpenMPToLLVMIRTranslation
  AIIRTargetLLVMIRExport
  )

if(LLVM_BUILD_LLVM_DYLIB AND NOT (WIN32 OR MINGW OR CYGWIN)) # Does not build on windows currently, see #106859
  # Build a shared library for the execution engine. Some downstream projects
  # use this library to build their own CPU runners while preserving dynamic
  # linkage.
  add_aiir_library(AIIRExecutionEngineShared
    ExecutionEngine.cpp
    SHARED

    EXCLUDE_FROM_LIBAIIR

    ADDITIONAL_HEADER_DIRS
    ${AIIR_MAIN_INCLUDE_DIR}/aiir/ExecutionEngine

    # Ensures that all necessary dependencies are resolved.
    DEPENDS
    AIIRExecutionEngine

    LINK_LIBS PUBLIC
    LLVM
    AIIR
    )
endif()

get_property(dialect_libs GLOBAL PROPERTY AIIR_DIALECT_LIBS)
add_aiir_library(AIIRJitRunner
  JitRunner.cpp

  EXCLUDE_FROM_LIBAIIR

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  OrcJIT
  JITLink

  LINK_LIBS PUBLIC
  AIIRExecutionEngine
)
aiir_target_link_libraries(AIIRJitRunner PUBLIC
  ${dialect_libs}
  AIIRFuncDialect
  AIIRFuncToLLVM
  AIIRIR
  AIIRParser
  AIIRLLVMToLLVMIRTranslation
  AIIRTargetLLVMIRExport
  AIIRTransforms
  AIIRSupport
)

# When -fPIC is not provided, shared libraries cannot be built if it links against
# non-PIC code.
if(LLVM_ENABLE_PIC)
  add_aiir_library(aiir_float16_utils
    SHARED
    Float16bits.cpp

    EXCLUDE_FROM_LIBAIIR

    STANDALONE
    )
  set_property(TARGET aiir_float16_utils PROPERTY CXX_STANDARD 17)
  target_compile_definitions(aiir_float16_utils PRIVATE aiir_float16_utils_EXPORTS)

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    # TODO: This support library is only used on Linux and Mac builds until we figure
    # out how to hide LLVM symbols in a way that works for all platforms.
    add_aiir_library(aiir_apfloat_wrappers
      SHARED
      APFloatWrappers.cpp

      EXCLUDE_FROM_LIBAIIR

      # Disable PCH reuse due to non-default symbol visibility.
      DISABLE_PCH_REUSE
      )
    set_target_properties(
      aiir_apfloat_wrappers
      PROPERTIES CXX_STANDARD 17
                CXX_VISIBILITY_PRESET hidden
                VISIBILITY_INLINES_HIDDEN ON
    )
    target_compile_definitions(aiir_apfloat_wrappers PRIVATE aiir_apfloat_wrappers_EXPORTS)
    # Hide LLVM symbols to avoid ODR violations.
    target_link_options(aiir_apfloat_wrappers PRIVATE $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>)
  endif()

  add_subdirectory(SparseTensor)

  add_aiir_library(aiir_c_runner_utils
    SHARED
    CRunnerUtils.cpp
    SparseTensorRuntime.cpp

    EXCLUDE_FROM_LIBAIIR

    LINK_LIBS PUBLIC
    aiir_float16_utils
    AIIRSparseTensorEnums
    AIIRSparseTensorRuntime

    STANDALONE
    )
  set_property(TARGET aiir_c_runner_utils PROPERTY CXX_STANDARD 17)
  target_compile_definitions(aiir_c_runner_utils PRIVATE aiir_c_runner_utils_EXPORTS)

  # Conditionally link apfloat wrappers only on Linux and Mac.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(aiir_c_runner_utils PUBLIC aiir_apfloat_wrappers)
  endif()

  add_aiir_library(aiir_runner_utils
    SHARED
    RunnerUtils.cpp

    EXCLUDE_FROM_LIBAIIR

    LINK_LIBS PUBLIC
    aiir_float16_utils

    STANDALONE
  )
  target_compile_definitions(aiir_runner_utils PRIVATE aiir_runner_utils_EXPORTS)

    # Conditionally link apfloat wrappers only on Linux.
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(aiir_runner_utils PUBLIC aiir_apfloat_wrappers)
  endif()

  add_aiir_library(aiir_async_runtime
    SHARED
    AsyncRuntime.cpp

    EXCLUDE_FROM_LIBAIIR

    LINK_LIBS PUBLIC
    ${LLVM_PTHREAD_LIB}

    # TODO: this is merely a workaround. If this library depends on LLVMSupport,
    # it should suppress symbols, or if it doesn't, it shouldn't link against
    # it. This workaround prevents the library from defining the symbol
    # llvm::EnableABIBreakingChecks, which would cause ODR-violations when
    # dlopen-ed.
    DISABLE_PCH_REUSE
  )
  set_property(TARGET aiir_async_runtime PROPERTY CXX_VISIBILITY_PRESET hidden)
  target_compile_definitions(aiir_async_runtime PRIVATE aiir_async_runtime_EXPORTS)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Don't export symbols from link-time dependencies, these are internal
    # implementation details.
    # FIXME: Add a similar fix for Windows.
    target_link_options(aiir_async_runtime PRIVATE "-Wl,-exclude-libs,ALL")
  endif()

  add_aiir_library(aiir_arm_sme_abi_stubs
    SHARED
    ArmSMEStubs.cpp

    # TODO: this is merely a workaround. If this library depends on LLVMSupport,
    # it should suppress symbols, or if it doesn't, it shouldn't link against
    # it. This workaround prevents the library from defining the symbol
    # llvm::EnableABIBreakingChecks, which would cause ODR-violations when
    # dlopen-ed.
    DISABLE_PCH_REUSE
  )
  target_compile_definitions(aiir_arm_sme_abi_stubs PRIVATE aiir_arm_sme_abi_stubs_EXPORTS)

  add_aiir_library(aiir_arm_runner_utils
    SHARED
    ArmRunnerUtils.cpp

    # TODO: this is merely a workaround. If this library depends on LLVMSupport,
    # it should suppress symbols, or if it doesn't, it shouldn't link against
    # it. This workaround prevents the library from defining the symbol
    # llvm::EnableABIBreakingChecks, which would cause ODR-violations when
    # dlopen-ed.
    DISABLE_PCH_REUSE
  )

  if(AIIR_ENABLE_CUDA_RUNNER)
    # Configure CUDA support. Using check_language first allows us to give a
    # custom error message.
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
    else()
      message(SEND_ERROR
        "Building the aiir cuda runner requires a working CUDA install")
    endif()

    # We need the libcuda.so library.
    find_library(CUDA_RUNTIME_LIBRARY cuda HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} REQUIRED)

    add_aiir_library(aiir_cuda_runtime
      SHARED
      CudaRuntimeWrappers.cpp

      EXCLUDE_FROM_LIBAIIR

      STANDALONE
    )
    set_property(TARGET aiir_cuda_runtime PROPERTY CXX_STANDARD 14)

    target_include_directories(aiir_cuda_runtime
      PRIVATE
      ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(aiir_cuda_runtime
      PRIVATE
      ${CUDA_RUNTIME_LIBRARY}
    )

    if(AIIR_ENABLE_CUDA_CUSPARSE)
      # Find the libcusparse.so library if CUSPARSE build is requested.
      find_library(CUDA_CUSPARSE_LIBRARY cusparse HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} REQUIRED)

      target_link_libraries(aiir_cuda_runtime
        PRIVATE
        ${CUDA_CUSPARSE_LIBRARY}
      )
      target_compile_definitions(aiir_cuda_runtime
        PRIVATE
        AIIR_ENABLE_CUDA_CUSPARSE=1
      )

      if(AIIR_ENABLE_CUDA_CUSPARSELT)
        # Find the libcusparseLt.so library in package manager default path if
        # CUSPARSELT build is requested. libcusparseLt.so provides sm80+ tensor
        # core support for 2:4 sparsity acceleration.
        find_library(CUDA_CUSPARSELT_LIBRARY cusparseLt HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} REQUIRED)
        find_path(CUDA_CUSPARSELT_HEADER cusparseLt.h HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} REQUIRED)
        target_include_directories(aiir_cuda_runtime
          PRIVATE
          ${CUDA_CUSPARSELT_HEADER}
        )
        target_link_libraries(aiir_cuda_runtime
          PRIVATE
          ${CUDA_CUSPARSELT_LIBRARY}
        )
        target_compile_definitions(aiir_cuda_runtime
          PRIVATE
          AIIR_ENABLE_CUDA_CUSPARSELT=1
        )
      endif()
    endif()
  endif()

  if(AIIR_ENABLE_ROCM_RUNNER)
    # Configure ROCm support.
    if (NOT DEFINED ROCM_PATH)
      if (NOT DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to which ROCm has been installed")
      else()
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Path to which ROCm has been installed")
      endif()
    endif()
    # A lot of the ROCm CMake files expect to find their own dependencies in
    # CMAKE_PREFIX_PATH and don't respect PATHS or HINTS :( .
    # Therefore, temporarily add the ROCm path to CMAKE_PREFIX_PATH so we can
    # load HIP, then remove it
    set(REAL_CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} "${ROCM_PATH}/hip")
    find_package(hip REQUIRED)
    set(CMAKE_PREFIX_PATH "${REAL_CMAKE_PREFIX_PATH}")

    if (NOT DEFINED ROCM_TEST_CHIPSET)
      find_program(ROCM_AGENT_ENUMERATOR rocm_agent_enumerator "${ROCM_PATH}/bin" /usr/bin /usr/local/bin)
      if(ROCM_AGENT_ENUMERATOR)
          execute_process(COMMAND "${ROCM_AGENT_ENUMERATOR}"
          OUTPUT_VARIABLE AGENTS_STRING
          ERROR_VARIABLE AGENTS_STRING
          RESULT_VARIABLE AGENT_ENUMERATOR_RESULT)
      else()
        message(SEND_ERROR "Could not find rocm_agent_enumerator")
      endif()
      if (NOT AGENT_ENUMERATOR_RESULT EQUAL 0)
        message(SEND_ERROR "Could not run rocm_agent_enumerator and ROCM_TEST_CHIPSET is not defined")
        set(AGENTS_STRING "")
      endif()
      string(STRIP AGENTS_STRING ${AGENTS_STRING})
      string(REPLACE "\n" ";" AGENTS_LIST ${AGENTS_STRING})
      if (AGENTS_LIST STREQUAL "")
        message(SEND_ERROR "No non-CPU ROCm agents found on the system, and ROCM_TEST_CHIPSET is not defined")
      else()
        list(GET AGENTS_LIST 0 FIRST_AGENT)
        set(ROCM_TEST_CHIPSET ${FIRST_AGENT} CACHE STRING "Chipset for which to compile ROCm integration tests")
        message(STATUS "Compiling integration tests for ${ROCM_TEST_CHIPSET}")
      endif()
    endif()

    add_aiir_library(aiir_rocm_runtime
      SHARED
      RocmRuntimeWrappers.cpp

      EXCLUDE_FROM_LIBAIIR

      # TODO: this is merely a workaround. If this library depends on LLVMSupport,
      # it should suppress symbols, or if it doesn't, it shouldn't link against
      # it. This workaround prevents the library from defining the symbol
      # llvm::EnableABIBreakingChecks, which would cause ODR-violations when
      # dlopen-ed.
      DISABLE_PCH_REUSE
    )

    # Supress compiler warnings from HIP headers
    check_cxx_compiler_flag(-Wno-c++98-compat-extra-semi
      CXX_SUPPORTS_NO_CXX98_COMPAT_EXTRA_SEMI_FLAG)
    if (CXX_SUPPORTS_CXX98_COMPAT_EXTRA_SEMI_FLAG)
      target_compile_options(aiir_rocm_runtime PRIVATE
        "-Wno-c++98-compat-extra-semi")
    endif()
    check_cxx_compiler_flag(-Wno-return-type-c-linkage
        CXX_SUPPORTS_WNO_RETURN_TYPE_C_LINKAGE_FLAG)
    if (CXX_SUPPORTS_WNO_RETURN_TYPE_C_LINKAGE_FLAG)
      target_compile_options(aiir_rocm_runtime PRIVATE
        "-Wno-return-type-c-linkage")
    endif()
    check_cxx_compiler_flag(-Wno-nested-anon-types
      CXX_SUPPORTS_WNO_NESTED_ANON_TYPES_FLAG)
    if (CXX_SUPPORTS_WNO_NESTED_ANON_TYPES_FLAG)
      target_compile_options(aiir_rocm_runtime PRIVATE
        "-Wno-nested-anon-types")
    endif()
    check_cxx_compiler_flag(-Wno-gnu-anonymous-struct
      CXX_SUPPORTS_WNO_GNU_ANONYMOUS_STRUCT_FLAG)
    if (CXX_SUPPORTS_WNO_GNU_ANONYMOUS_STRUCT_FLAG)
      target_compile_options(aiir_rocm_runtime PRIVATE
      "-Wno-gnu-anonymous-struct")
    endif()

    set_property(TARGET aiir_rocm_runtime
      PROPERTY INSTALL_RPATH_USE_LINK_PATH ON)

    target_link_libraries(aiir_rocm_runtime
      PUBLIC
      hip::host hip::amdhip64
    )
  endif()

  if(AIIR_ENABLE_SYCL_RUNNER OR AIIR_ENABLE_LEVELZERO_RUNNER)
    # Both runtimes require LevelZero, so we can find it once.
    find_package(LevelZeroRuntime)

    if(NOT LevelZeroRuntime_FOUND)
      message(FATAL_ERROR "LevelZero not found. Please set LEVEL_ZERO_DIR.")
    endif()
  endif()

  if(AIIR_ENABLE_SYCL_RUNNER)
    find_package(SyclRuntime)

    if(NOT SyclRuntime_FOUND)
      message(FATAL_ERROR "syclRuntime not found. Please set check oneapi installation and run setvars.sh.")
    endif()

    set(LLVM_REQUIRES_RTTI ON)
    set(LLVM_REQUIRES_EH ON)
    add_aiir_library(aiir_sycl_runtime
      SHARED
      SyclRuntimeWrappers.cpp

      EXCLUDE_FROM_LIBAIIR

      STANDALONE
    )
    unset(LLVM_REQUIRES_RTTI)
    unset(LLVM_REQUIRES_EH)


    target_include_directories(aiir_sycl_runtime PRIVATE
      ${AIIR_INCLUDE_DIRS}
    )

    target_link_libraries(aiir_sycl_runtime PRIVATE LevelZeroRuntime::LevelZeroRuntime SyclRuntime::SyclRuntime)

    set_property(TARGET aiir_sycl_runtime APPEND PROPERTY BUILD_RPATH "${LevelZeroRuntime_LIBRARIES_DIR}" "${SyclRuntime_LIBRARIES_DIR}")
  endif()

  if(AIIR_ENABLE_LEVELZERO_RUNNER)
    set(LLVM_REQUIRES_RTTI ON)
    set(LLVM_REQUIRES_EH ON)
    add_aiir_library(aiir_levelzero_runtime
      SHARED
      LevelZeroRuntimeWrappers.cpp

      EXCLUDE_FROM_LIBAIIR

      STANDALONE
    )
    unset(LLVM_REQUIRES_RTTI)
    unset(LLVM_REQUIRES_EH)

    target_include_directories(aiir_levelzero_runtime PRIVATE
      ${AIIR_INCLUDE_DIRS}
    )

    target_link_libraries(aiir_levelzero_runtime PRIVATE LevelZeroRuntime::LevelZeroRuntime)

    set_property(TARGET aiir_levelzero_runtime APPEND PROPERTY BUILD_RPATH "${LevelZeroRuntime_LIBRARIES_DIR}")
  endif()

  if(AIIR_ENABLE_SPIRV_CPU_RUNNER)
    add_aiir_library(aiir_spirv_cpu_runtime
      SHARED
      SpirvCpuRuntimeWrappers.cpp

      EXCLUDE_FROM_LIBAIIR

      STANDALONE
    )

    target_compile_definitions(aiir_spirv_cpu_runtime
      PRIVATE
      aiir_spirv_cpu_runtime_EXPORTS)
  endif()

  if (AIIR_ENABLE_VULKAN_RUNNER)
    find_package(Vulkan)

    # If Vulkan is not found try a path specified by VULKAN_SDK.
    if (NOT Vulkan_FOUND)
      if ("$ENV{VULKAN_SDK}" STREQUAL "")
        message(FATAL_ERROR "Vulkan not found through CMake; please provide "
                            "VULKAN_SDK path as an environment variable")
      endif()

      find_library(Vulkan_LIBRARY vulkan HINTS "$ENV{VULKAN_SDK}/lib" REQUIRED)
      if (Vulkan_LIBRARY)
        set(Vulkan_FOUND ON)
        set(Vulkan_INCLUDE_DIR "$ENV{VULKAN_SDK}/include")
        message(STATUS "Found Vulkan: " ${Vulkan_LIBRARY})
      endif()
    endif()

    if (NOT Vulkan_FOUND)
      message(FATAL_ERROR "Cannot find Vulkan library")
    endif()

    add_aiir_library(aiir_vulkan_runtime SHARED
      VulkanRuntimeWrappers.cpp
      VulkanRuntime.cpp

      EXCLUDE_FROM_LIBAIIR

      STANDALONE
    )

    target_include_directories(aiir_vulkan_runtime
      PUBLIC
      ${Vulkan_INCLUDE_DIR}
    )

    # *IMPORTANT*: This library cannot depend on LLVM libraries. Otherwise,
    # it may cause LLVM version conflict when used together with other shared
    # libraries depending on LLVM. Notably, Mesa, who implements Vulkan
    # drivers on Linux, depends on the system libLLVM.so.
    target_link_libraries(aiir_vulkan_runtime
      PUBLIC
      ${Vulkan_LIBRARY}
    )
  endif()
endif()
