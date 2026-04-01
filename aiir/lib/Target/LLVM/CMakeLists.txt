add_aiir_library(AIIRTargetLLVM
  ModuleToObject.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Target/LLVM

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  BitWriter
  Core
  IPO
  IRReader
  Linker
  MC
  Passes
  Support
  Target
  LINK_LIBS PUBLIC
  AIIRExecutionEngineUtils
  AIIRTargetLLVMIRExport
)

if ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(NVPTX_LIBS
    NVPTXCodeGen
    NVPTXDesc
    NVPTXInfo
  )
endif()

add_aiir_dialect_library(AIIRNVVMTarget
  NVVM/Target.cpp

  OBJECT

  ADDITIONAL_HEADER_DIRS
  ${AIIR_MAIN_INCLUDE_DIR}/aiir/Dialect/LLVMIR

  LINK_COMPONENTS
  ${NVPTX_LIBS}

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRExecutionEngineUtils
  AIIRSupport
  AIIRGPUDialect
  AIIRTargetLLVM
  AIIRNVVMToLLVMIRTranslation
  )

if ("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  # Find the CUDA toolkit.
  find_package(CUDAToolkit)

  if(CUDAToolkit_FOUND)
    # Get the CUDA toolkit path. The path is needed for detecting `libdevice.bc`.
    # These extra steps are needed because of a bug on CMake.
    # See: https://gitlab.kitware.com/cmake/cmake/-/issues/24858
    # TODO: Bump the AIIR CMake version to 3.26.4 and switch to
    # ${CUDAToolkit_LIBRARY_ROOT}
    if(NOT DEFINED CUDAToolkit_LIBRARY_ROOT)
      get_filename_component(AIIR_CUDAToolkit_ROOT ${CUDAToolkit_BIN_DIR}
                             DIRECTORY ABSOLUTE)
    else()
      set(AIIR_CUDAToolkit_ROOT ${CUDAToolkit_LIBRARY_ROOT})
    endif()

    # Add the `nvptxcompiler` library.
    if(AIIR_ENABLE_NVPTXCOMPILER)
      # Find the `nvptxcompiler` library.
      # TODO: Bump the AIIR CMake version to 3.25 and use `CUDA::nvptxcompiler_static`.
      find_library(AIIR_NVPTXCOMPILER_LIB_PATH nvptxcompiler_static
                  PATHS ${CUDAToolkit_LIBRARY_DIR} NO_DEFAULT_PATH)

      # Fail if `nvptxcompiler_static` couldn't be found.
      if(AIIR_NVPTXCOMPILER_LIB_PATH STREQUAL "AIIR_NVPTXCOMPILER_LIB_PATH-NOTFOUND")
        message(FATAL_ERROR
                "Requested using the `nvptxcompiler` library backend but it couldn't be found.")
      endif()

      add_library(AIIR_NVPTXCOMPILER_LIB STATIC IMPORTED GLOBAL)
      # Downstream projects can modify this path and use it in CMake. For example:
      # add_library(AIIR_NVPTXCOMPILER_LIB STATIC IMPORTED GLOBAL)
      # set_property(TARGET AIIR_NVPTXCOMPILER_LIB PROPERTY IMPORTED_LOCATION ${...})  
      # where `...` is to be replaced with the path to the library.
      set_property(TARGET AIIR_NVPTXCOMPILER_LIB PROPERTY IMPORTED_LOCATION ${AIIR_NVPTXCOMPILER_LIB_PATH})
      # Link against `nvptxcompiler_static`. TODO: use `CUDA::nvptxcompiler_static`.
      target_link_libraries(AIIRNVVMTarget PRIVATE AIIR_NVPTXCOMPILER_LIB)
      target_include_directories(obj.AIIRNVVMTarget PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

      # Add the `nvfatbin` library.
      find_library(AIIR_NVFATBIN_LIB_PATH nvfatbin_static
                  PATHS ${CUDAToolkit_LIBRARY_DIR} NO_DEFAULT_PATH)
      # Fail if `nvfatbin_static` couldn't be found.
      if(AIIR_NVFATBIN_LIB_PATH STREQUAL "AIIR_NVFATBIN_LIB_PATH-NOTFOUND")
        message(FATAL_ERROR
                "Requested using the static `nvptxcompiler` library which requires the \
                'nvfatbin` library, but it couldn't be found.")
      endif()

      add_library(AIIR_NVFATBIN_LIB STATIC IMPORTED GLOBAL)
      set_property(TARGET AIIR_NVFATBIN_LIB PROPERTY IMPORTED_LOCATION ${AIIR_NVFATBIN_LIB_PATH})  
      target_link_libraries(AIIRNVVMTarget PRIVATE AIIR_NVFATBIN_LIB)
    endif()
  else()
    # Fail if `AIIR_ENABLE_NVPTXCOMPILER` is enabled and the toolkit couldn't be found.
    if(AIIR_ENABLE_NVPTXCOMPILER)
      message(FATAL_ERROR
              "Requested using the `nvptxcompiler` library backend but it couldn't be found.")
    endif()
  endif()
  message(VERBOSE "AIIR default CUDA toolkit path: ${AIIR_CUDAToolkit_ROOT}")

  # Define the `CUDAToolkit` path.
  target_compile_definitions(obj.AIIRNVVMTarget
    PRIVATE
    __DEFAULT_CUDATOOLKIT_PATH__="${AIIR_CUDAToolkit_ROOT}"
  )
endif()


function(embed_binary_to_src file output_file symbol)
    file(READ ${file} filedata HEX)
    # Convert hex data for C compatibility
    string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," filedata ${filedata})
    # Write data to output file
    file(WRITE ${output_file} "const unsigned char ${symbol}[] = {${filedata}};\nconst int ${symbol}_size = sizeof(${symbol});\n")
endfunction()

set(AIIR_NVVM_EMBED_LIBDEVICE 0 CACHE BOOL "Embed CUDA libdevice.bc in the binary at build time instead of looking it up at runtime")
if (AIIR_NVVM_EMBED_LIBDEVICE)
  if (NOT AIIR_NVVM_LIBDEVICE_PATH)
    if(CUDAToolkit_FOUND)
      find_file(AIIR_NVVM_LIBDEVICE_PATH libdevice.10.bc
                PATHS ${CUDAToolkit_LIBRARY_ROOT}
                PATH_SUFFIXES "nvvm/libdevice" NO_DEFAULT_PATH REQUIRED)
    else()
      message(FATAL_ERROR
              "Requested using the `nvptxcompiler` library backend but it couldn't be found.")
    endif()
  endif()
  
  embed_binary_to_src(${AIIR_NVVM_LIBDEVICE_PATH} ${CMAKE_CURRENT_BINARY_DIR}/libdevice_embedded.c _aiir_embedded_libdevice)
  add_aiir_library(AIIRNVVMLibdevice
    ${CMAKE_CURRENT_BINARY_DIR}/libdevice_embedded.c
  )
  target_link_libraries(AIIRNVVMTarget PRIVATE AIIRNVVMLibdevice)
  target_compile_definitions(obj.AIIRNVVMTarget
    PRIVATE
    AIIR_NVVM_EMBED_LIBDEVICE=1
  )
else()
  target_compile_definitions(obj.AIIRNVVMTarget
    PRIVATE
    AIIR_NVVM_EMBED_LIBDEVICE=0
  )
endif()


if ("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(AMDGPU_LIBS
    AMDGPUAsmParser
    AMDGPUCodeGen
    AMDGPUDesc
    AMDGPUInfo
  )
endif()

add_aiir_dialect_library(AIIRROCDLTarget
  ROCDL/Target.cpp
  ROCDL/Utils.cpp

  OBJECT

  LINK_COMPONENTS
  FrontendOffloading
  MCParser
  ${AMDGPU_LIBS}

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRExecutionEngineUtils
  AIIRSupport
  AIIRGPUDialect
  AIIRTargetLLVM
  AIIRROCDLToLLVMIRTranslation
  )

if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  if (DEFINED ROCM_PATH)
    set(DEFAULT_ROCM_PATH "${ROCM_PATH}" CACHE PATH "Fallback path to search for ROCm installs")
  elseif(DEFINED ENV{ROCM_PATH})
    set(DEFAULT_ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "Fallback path to search for ROCm installs")
  else()
    if (WIN32)
      # Avoid setting an UNIX path for Windows.
      # TODO: Eventually migrate to FindHIP once it becomes a part of CMake.
      set(DEFAULT_ROCM_PATH "" CACHE PATH "Fallback path to search for ROCm installs")
    else()
      set(DEFAULT_ROCM_PATH "/opt/rocm" CACHE PATH "Fallback path to search for ROCm installs")
    endif()
  endif()
  message(VERBOSE "AIIR Default ROCM toolkit path: ${DEFAULT_ROCM_PATH}")

  target_compile_definitions(obj.AIIRROCDLTarget
    PRIVATE
    __DEFAULT_ROCM_PATH__="${DEFAULT_ROCM_PATH}"
  )
endif()

if ("SPIRV" IN_LIST LLVM_TARGETS_TO_BUILD)
  set(SPIRV_LIBS
    SPIRVCodeGen
    SPIRVDesc
    SPIRVInfo
  )
endif()

add_aiir_dialect_library(AIIRXeVMTarget
  XeVM/Target.cpp

  OBJECT

  LINK_COMPONENTS
  ${SPIRV_LIBS}

  LINK_LIBS PUBLIC
  AIIRIR
  AIIRExecutionEngineUtils
  AIIRSupport
  AIIRGPUDialect
  AIIRTargetLLVM
  AIIRXeVMToLLVMIRTranslation
)
