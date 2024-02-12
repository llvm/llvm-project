set(OBJECT_LIBRARY_TARGET_TYPE "OBJECT_LIBRARY")

function(_get_compile_options_from_flags output_var)
  set(compile_options "")

  if(LIBC_TARGET_ARCHITECTURE_IS_RISCV64 OR(LIBC_CPU_FEATURES MATCHES "FMA"))
    check_flag(ADD_FMA_FLAG ${FMA_OPT_FLAG} ${flags})
  endif()
  check_flag(ADD_SSE4_2_FLAG ${ROUND_OPT_FLAG} ${flags})
  check_flag(ADD_EXPLICIT_SIMD_OPT_FLAG ${EXPLICIT_SIMD_OPT_FLAG} ${flags})
  
  if(LLVM_COMPILER_IS_GCC_COMPATIBLE)
    if(ADD_FMA_FLAG)
      if(LIBC_TARGET_ARCHITECTURE_IS_X86)
        list(APPEND compile_options "-mavx2")
        list(APPEND compile_options "-mfma")
      elseif(LIBC_TARGET_ARCHITECTURE_IS_RISCV64)
        list(APPEND compile_options "-D__LIBC_RISCV_USE_FMA")
      endif()
    endif()
    if(ADD_SSE4_2_FLAG)
      list(APPEND compile_options "-msse4.2")
    endif()
    if(ADD_EXPLICIT_SIMD_OPT_FLAG)
      list(APPEND compile_options "-D__LIBC_EXPLICIT_SIMD_OPT")
    endif()
  elseif(MSVC)
    if(ADD_FMA_FLAG)
      list(APPEND compile_options "/arch:AVX2")
    endif()
    if(ADD_EXPLICIT_SIMD_OPT_FLAG)
      list(APPEND compile_options "/D__LIBC_EXPLICIT_SIMD_OPT")
    endif()
  endif()

  set(${output_var} ${compile_options} PARENT_SCOPE)
endfunction(_get_compile_options_from_flags)

function(_get_common_compile_options output_var flags)
  _get_compile_options_from_flags(compile_flags ${flags})

  set(compile_options ${LIBC_COMPILE_OPTIONS_DEFAULT} ${compile_flags})

  if(LLVM_COMPILER_IS_GCC_COMPATIBLE)
    list(APPEND compile_options "-fpie")

    if(LLVM_LIBC_FULL_BUILD)
      # Only add -ffreestanding flag in full build mode.
      list(APPEND compile_options "-ffreestanding")
    endif()

    list(APPEND compile_options "-fno-builtin")
    list(APPEND compile_options "-fno-exceptions")
    list(APPEND compile_options "-fno-lax-vector-conversions")
    list(APPEND compile_options "-fno-unwind-tables")
    list(APPEND compile_options "-fno-asynchronous-unwind-tables")
    list(APPEND compile_options "-fno-rtti")
    if (LIBC_CC_SUPPORTS_PATTERN_INIT)
      list(APPEND compile_options "-ftrivial-auto-var-init=pattern")
    endif()
    list(APPEND compile_options "-Wall")
    list(APPEND compile_options "-Wextra")
    # -DLIBC_WNO_ERROR=ON if you can't build cleanly with -Werror.
    if(NOT LIBC_WNO_ERROR)
      list(APPEND compile_options "-Werror")
    endif()
    list(APPEND compile_options "-Wconversion")
    list(APPEND compile_options "-Wno-sign-conversion")
    list(APPEND compile_options "-Wimplicit-fallthrough")
    list(APPEND compile_options "-Wwrite-strings")
    list(APPEND compile_options "-Wextra-semi")
    if(NOT CMAKE_COMPILER_IS_GNUCXX)
      list(APPEND compile_options "-Wnewline-eof")
      list(APPEND compile_options "-Wnonportable-system-include-path")
      list(APPEND compile_options "-Wstrict-prototypes")
      list(APPEND compile_options "-Wthread-safety")
      list(APPEND compile_options "-Wglobal-constructors")
    endif()
  elseif(MSVC)
    list(APPEND compile_options "/EHs-c-")
    list(APPEND compile_options "/GR-")
  endif()
  if (LIBC_TARGET_ARCHITECTURE_IS_GPU)
    list(APPEND compile_options "-nogpulib")
    list(APPEND compile_options "-fvisibility=hidden")
    list(APPEND compile_options "-fconvergent-functions")

    # Manually disable all standard include paths and include the resource
    # directory to prevent system headers from being included.
    list(APPEND compile_options "-isystem${COMPILER_RESOURCE_DIR}/include")
    list(APPEND compile_options "-nostdinc")
  endif()
  set(${output_var} ${compile_options} PARENT_SCOPE)
endfunction()

# Obtains NVPTX specific arguments for compilation.
# The PTX feature is primarily based on the CUDA toolchain version. We want to
# be able to target NVPTX without an existing CUDA installation, so we need to
# set this manually. This simply sets the PTX feature to the minimum required
# for the features we wish to use on that target. The minimum PTX features used
# here roughly corresponds to the CUDA 9.0 release.
# Adjust as needed for desired PTX features.
function(get_nvptx_compile_options output_var gpu_arch)
  set(nvptx_options "")
  list(APPEND nvptx_options "-march=${gpu_arch}")
  list(APPEND nvptx_options "-Wno-unknown-cuda-version")
  list(APPEND nvptx_options "SHELL:-mllvm -nvptx-emit-init-fini-kernel=false")
  if(${gpu_arch} STREQUAL "sm_35")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_37")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_50")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_52")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_53")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_60")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_61")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_62")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_70")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_72")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_75")
    list(APPEND nvptx_options "--cuda-feature=+ptx63")
  elseif(${gpu_arch} STREQUAL "sm_80")
    list(APPEND nvptx_options "--cuda-feature=+ptx72")
  elseif(${gpu_arch} STREQUAL "sm_86")
    list(APPEND nvptx_options "--cuda-feature=+ptx72")
  elseif(${gpu_arch} STREQUAL "sm_89")
    list(APPEND nvptx_options "--cuda-feature=+ptx72")
  elseif(${gpu_arch} STREQUAL "sm_90")
    list(APPEND nvptx_options "--cuda-feature=+ptx72")
  else()
    message(FATAL_ERROR "Unknown Nvidia GPU architecture '${gpu_arch}'")
  endif()

  if(LIBC_CUDA_ROOT)
    list(APPEND nvptx_options "--cuda-path=${LIBC_CUDA_ROOT}")
  endif()
  set(${output_var} ${nvptx_options} PARENT_SCOPE)
endfunction()

# Build the object target for a single GPU arch.
# Usage:
#     _build_gpu_object_for_single_arch(
#       <target_name>
#       <gpu_arch>
#       SRCS <list of .cpp files>
#       HDRS <list of .h files>
#       DEPENDS <list of dependencies>
#       COMPILE_OPTIONS <optional list of special compile options for this target>
#       FLAGS <optional list of flags>
#     )
function(_build_gpu_object_for_single_arch fq_target_name gpu_arch)
  cmake_parse_arguments(
    "ADD_GPU_OBJ"
    "" # No optional arguments
    "NAME;CXX_STANDARD" # Single value arguments
    "SRCS;HDRS;DEPENDS;COMPILE_OPTIONS;FLAGS"  # Multi value arguments
    ${ARGN}
  )

  if(NOT ADD_GPU_OBJ_CXX_STANDARD)
    set(ADD_GPU_OBJ_CXX_STANDARD ${CMAKE_CXX_STANDARD})
  endif()

  set(compile_options ${ADD_GPU_OBJ_COMPILE_OPTIONS})
  # Derive the triple from the specified architecture.
  if("${gpu_arch}" IN_LIST all_amdgpu_architectures)
    set(gpu_target_triple ${AMDGPU_TARGET_TRIPLE})
    list(APPEND compile_options "-mcpu=${gpu_arch}")
    list(APPEND compile_options "SHELL:-Xclang -mcode-object-version=none")
    list(APPEND compile_options "-emit-llvm")
  elseif("${gpu_arch}" IN_LIST all_nvptx_architectures)
    set(gpu_target_triple ${NVPTX_TARGET_TRIPLE})
    get_nvptx_compile_options(nvptx_options ${gpu_arch})
    list(APPEND compile_options "${nvptx_options}")
  else()
    message(FATAL_ERROR "Unknown GPU architecture '${gpu_arch}'")
  endif()
  list(APPEND compile_options "--target=${gpu_target_triple}")

  # Build the library for this target architecture. We always emit LLVM-IR for
  # packaged GPU binaries.
  add_library(${fq_target_name}
    EXCLUDE_FROM_ALL
    OBJECT
    ${ADD_GPU_OBJ_SRCS}
    ${ADD_GPU_OBJ_HDRS}
  )

  target_compile_options(${fq_target_name} PRIVATE ${compile_options})
  target_include_directories(${fq_target_name} SYSTEM PRIVATE ${LIBC_INCLUDE_DIR})
  target_include_directories(${fq_target_name} PRIVATE ${LIBC_SOURCE_DIR})
  set_target_properties(${fq_target_name} PROPERTIES CXX_STANDARD ${ADD_GPU_OBJ_CXX_STANDARD})
  if(ADD_GPU_OBJ_DEPENDS)
    add_dependencies(${fq_target_name} ${ADD_GPU_OBJ_DEPENDS})
    set_target_properties(${fq_target_name} PROPERTIES DEPS "${ADD_GPU_OBJ_DEPENDS}")
  endif()
endfunction(_build_gpu_object_for_single_arch)

# Build the object target for the GPU.
# This compiles the target for all supported architectures and embeds it into
# host binary for installing.
# Usage:
#     _build_gpu_object_bundle(
#       <target_name>
#       SRCS <list of .cpp files>
#       HDRS <list of .h files>
#       DEPENDS <list of dependencies>
#       COMPILE_OPTIONS <optional list of special compile options for this target>
#       FLAGS <optional list of flags>
#     )
function(_build_gpu_object_bundle fq_target_name)
  cmake_parse_arguments(
    "ADD_GPU_OBJ"
    "" # No optional arguments
    "NAME;CXX_STANDARD" # Single value arguments
    "SRCS;HDRS;DEPENDS;COMPILE_OPTIONS;FLAGS"  # Multi value arguments
    ${ARGN}
  )

  if(NOT ADD_GPU_OBJ_CXX_STANDARD)
    set(ADD_GPU_OBJ_CXX_STANDARD ${CMAKE_CXX_STANDARD})
  endif()

  foreach(add_gpu_obj_src ${ADD_GPU_OBJ_SRCS})
    # The packaged version will be built for every target GPU architecture. We do
    # this so we can support multiple accelerators on the same machine.
    foreach(gpu_arch ${LIBC_GPU_ARCHITECTURES})
      get_filename_component(src_name ${add_gpu_obj_src} NAME)
      set(gpu_target_name ${fq_target_name}.${src_name}.${gpu_arch})

      _build_gpu_object_for_single_arch(
        ${gpu_target_name}
        ${gpu_arch}
        CXX_STANDARD ${ADD_GPU_OBJ_CXX_STANDARD}
        HDRS ${ADD_GPU_OBJ_HDRS}
        SRCS ${add_gpu_obj_src}
        COMPILE_OPTIONS
          ${ADD_GPU_OBJ_COMPILE_OPTIONS}
          "-emit-llvm"
        DEPENDS ${ADD_GPU_OBJ_DEPENDS}
      )
      # Append this target to a list of images to package into a single binary.
      set(input_file $<TARGET_OBJECTS:${gpu_target_name}>)
      if("${gpu_arch}" IN_LIST all_nvptx_architectures)
        get_nvptx_compile_options(nvptx_options ${gpu_arch})
        string(REGEX MATCH "\\+ptx[0-9]+" nvptx_ptx_feature ${nvptx_options})
        list(APPEND packager_images
             --image=file=${input_file},arch=${gpu_arch},triple=${NVPTX_TARGET_TRIPLE},feature=${nvptx_ptx_feature})
      else()
        list(APPEND packager_images
             --image=file=${input_file},arch=${gpu_arch},triple=${AMDGPU_TARGET_TRIPLE})
       endif()
      list(APPEND gpu_target_objects ${input_file})
    endforeach()

    # After building the target for the desired GPUs we must package the output
    # into a fatbinary, see https://clang.llvm.org/docs/OffloadingDesign.html for
    # more information.
    set(packaged_target_name ${fq_target_name}.${src_name}.__gpu__)
    set(packaged_output_name ${CMAKE_CURRENT_BINARY_DIR}/${fq_target_name}.${src_name}.gpubin)

    add_custom_command(OUTPUT ${packaged_output_name}
                       COMMAND ${LIBC_CLANG_OFFLOAD_PACKAGER}
                               ${packager_images} -o ${packaged_output_name}
                       DEPENDS ${gpu_target_objects} ${add_gpu_obj_src} ${ADD_GPU_OBJ_HDRS}
                       COMMENT "Packaging LLVM offloading binary")
    add_custom_target(${packaged_target_name} DEPENDS ${packaged_output_name})
    list(APPEND packaged_gpu_names ${packaged_target_name})
    list(APPEND packaged_gpu_binaries ${packaged_output_name})
  endforeach()

  # We create an empty 'stub' file for the host to contain the embedded device
  # code. This will be packaged into 'libcgpu.a'.
  # TODO: In the future we will want to combine every architecture for a target
  #       into a single bitcode file and use that. For now we simply build for
  #       every single one and let the offloading linker handle it.
  string(FIND ${fq_target_name} "." last_dot_loc REVERSE)
  math(EXPR name_loc "${last_dot_loc} + 1")
  string(SUBSTRING ${fq_target_name} ${name_loc} -1 target_name)
  set(stub_filename "${target_name}.cpp")
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/stubs/${stub_filename}"
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/stubs/
    COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/stubs/${stub_filename}
    DEPENDS ${gpu_target_objects} ${ADD_GPU_OBJ_SRCS} ${ADD_GPU_OBJ_HDRS}
  )
  set(stub_target_name ${fq_target_name}.__stub__)
  add_custom_target(${stub_target_name} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/stubs/${stub_filename})

  add_library(
    ${fq_target_name}
    # We want an object library as the objects will eventually get packaged into
    # an archive (like libcgpu.a).
    EXCLUDE_FROM_ALL
    OBJECT
    ${CMAKE_CURRENT_BINARY_DIR}/stubs/${stub_filename}
  )
  target_compile_options(${fq_target_name} BEFORE PRIVATE
                         ${ADD_GPU_OBJ_COMPILE_OPTIONS} -nostdlib)
  foreach(packaged_gpu_binary ${packaged_gpu_binaries})
    target_compile_options(${fq_target_name} PRIVATE
                           "SHELL:-Xclang -fembed-offload-object=${packaged_gpu_binary}")
  endforeach()
  target_include_directories(${fq_target_name} SYSTEM PRIVATE ${LIBC_INCLUDE_DIR})
  target_include_directories(${fq_target_name} PRIVATE ${LIBC_SOURCE_DIR})
  add_dependencies(${fq_target_name}
                   ${full_deps_list} ${packaged_gpu_names} ${stub_target_name})
endfunction()

# Rule which is essentially a wrapper over add_library to compile a set of
# sources to object files.
# Usage:
#     add_object_library(
#       <target_name>
#       HDRS <list of header files>
#       SRCS <list of source files>
#       [ALIAS] <If this object library is an alias for another object library.>
#       DEPENDS <list of dependencies; Should be a single item for ALIAS libraries>
#       COMPILE_OPTIONS <optional list of special compile options for this target>
#       FLAGS <optional list of flags>
function(create_object_library fq_target_name)
  cmake_parse_arguments(
    "ADD_OBJECT"
    "ALIAS;NO_GPU_BUNDLE" # optional arguments
    "CXX_STANDARD" # Single value arguments
    "SRCS;HDRS;COMPILE_OPTIONS;DEPENDS;FLAGS" # Multivalue arguments
    ${ARGN}
  )

  get_fq_deps_list(fq_deps_list ${ADD_OBJECT_DEPENDS})

  if(ADD_OBJECT_ALIAS)
    if(ADD_OBJECT_SRCS OR ADD_OBJECT_HDRS)
      message(FATAL_ERROR
              "${fq_target_name}: object library alias cannot have SRCS and/or HDRS.")
    endif()
    list(LENGTH fq_deps_list depends_size)
    if(NOT ${depends_size} EQUAL 1)
      message(FATAL_ERROR
              "${fq_targe_name}: object library alias should have exactly one DEPENDS.")
    endif()
    add_library(
      ${fq_target_name}
      ALIAS
      ${fq_deps_list}
    )
    return()
  endif()

  if(NOT ADD_OBJECT_SRCS)
    message(FATAL_ERROR "'add_object_library' rule requires SRCS to be specified.")
  endif()

  # The GPU build uses a separate internal file.
  if(LIBC_TARGET_ARCHITECTURE_IS_GPU AND NOT ${ADD_OBJECT_NO_GPU_BUNDLE})
    set(internal_target_name ${fq_target_name}.__internal__)
    set(public_packaging_for_internal "")
  else()
    set(internal_target_name ${fq_target_name})
    set(public_packaging_for_internal "-DLIBC_COPT_PUBLIC_PACKAGING")
  endif()

  _get_common_compile_options(compile_options "${ADD_OBJECT_FLAGS}")
  list(APPEND compile_options ${ADD_OBJECT_COMPILE_OPTIONS})

  # GPU builds require special handling for the objects because we want to
  # export several different targets at once, e.g. for both Nvidia and AMD.
  if(LIBC_TARGET_ARCHITECTURE_IS_GPU)
    if(NOT ${ADD_OBJECT_NO_GPU_BUNDLE})
      _build_gpu_object_bundle(
        ${fq_target_name}
        SRCS ${ADD_OBJECT_SRCS}
        HDRS ${ADD_OBJECT_HDRS}
        CXX_STANDARD ${ADD_OBJECT_CXX_STANDARD}
        COMPILE_OPTIONS ${compile_options} "-DLIBC_COPT_PUBLIC_PACKAGING"
        DEPENDS ${fq_deps_list}
      )
    endif()
    # When the target for GPU is not bundled, internal_target_name is the same
    # as fq_targetname
    _build_gpu_object_for_single_arch(
      ${internal_target_name}
      ${LIBC_GPU_TARGET_ARCHITECTURE}
      SRCS ${ADD_OBJECT_SRCS}
      HDRS ${ADD_OBJECT_HDRS}
      CXX_STANDARD ${ADD_OBJECT_CXX_STANDARD}
      COMPILE_OPTIONS ${compile_options} ${public_packaging_for_internal}
      DEPENDS ${fq_deps_list}
    )
  else()
    add_library(
      ${fq_target_name}
      EXCLUDE_FROM_ALL
      OBJECT
      ${ADD_OBJECT_SRCS}
      ${ADD_OBJECT_HDRS}
    )
    target_include_directories(${fq_target_name} SYSTEM PRIVATE ${LIBC_INCLUDE_DIR})
    target_include_directories(${fq_target_name} PRIVATE ${LIBC_SOURCE_DIR})
    target_compile_options(${fq_target_name} PRIVATE ${compile_options})
  endif()

  if(SHOW_INTERMEDIATE_OBJECTS)
    message(STATUS "Adding object library ${fq_target_name}")
    if(${SHOW_INTERMEDIATE_OBJECTS} STREQUAL "DEPS")
      foreach(dep IN LISTS ADD_OBJECT_DEPENDS)
        message(STATUS "  ${fq_target_name} depends on ${dep}")
      endforeach()
    endif()
  endif()

  if(fq_deps_list)
    add_dependencies(${fq_target_name} ${fq_deps_list})
    # Add deps as link libraries to inherit interface compile and link options.
    target_link_libraries(${fq_target_name} PUBLIC ${fq_deps_list})
  endif()

  if(NOT ADD_OBJECT_CXX_STANDARD)
    set(ADD_OBJECT_CXX_STANDARD ${CMAKE_CXX_STANDARD})
  endif()
  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      TARGET_TYPE ${OBJECT_LIBRARY_TARGET_TYPE}
      CXX_STANDARD ${ADD_OBJECT_CXX_STANDARD}
      DEPS "${fq_deps_list}"
      FLAGS "${ADD_OBJECT_FLAGS}"
  )

  if(TARGET ${internal_target_name})
    set_target_properties(
      ${fq_target_name}
      PROPERTIES
        OBJECT_FILES "$<TARGET_OBJECTS:${internal_target_name}>"
    )
  endif()
endfunction(create_object_library)

function(add_object_library target_name)
  add_target_with_flags(
    ${target_name}
    CREATE_TARGET create_object_library
    ${ARGN})
endfunction(add_object_library)

set(ENTRYPOINT_OBJ_TARGET_TYPE "ENTRYPOINT_OBJ")
set(ENTRYPOINT_OBJ_VENDOR_TARGET_TYPE "ENTRYPOINT_OBJ_VENDOR")

# A rule for entrypoint object targets.
# Usage:
#     add_entrypoint_object(
#       <target_name>
#       [ALIAS|REDIRECTED] # Specified if the entrypoint is redirected or an alias.
#       [NAME] <the C name of the entrypoint if different from target_name>
#       SRCS <list of .cpp files>
#       HDRS <list of .h files>
#       DEPENDS <list of dependencies>
#       COMPILE_OPTIONS <optional list of special compile options for this target>
#       SPECIAL_OBJECTS <optional list of special object targets added by the rule `add_object`>
#       FLAGS <optional list of flags>
#     )
function(create_entrypoint_object fq_target_name)
  cmake_parse_arguments(
    "ADD_ENTRYPOINT_OBJ"
    "ALIAS;REDIRECTED;VENDOR" # Optional argument
    "NAME;CXX_STANDARD" # Single value arguments
    "SRCS;HDRS;DEPENDS;COMPILE_OPTIONS;FLAGS"  # Multi value arguments
    ${ARGN}
  )

  set(entrypoint_target_type ${ENTRYPOINT_OBJ_TARGET_TYPE})
  if(${ADD_ENTRYPOINT_OBJ_VENDOR})
    # TODO: We currently rely on external definitions of certain math functions
    # provided by GPU vendors like AMD or Nvidia. We need to mark these so we
    # don't end up running tests on these. In the future all of these should be
    # implemented and this can be removed.
    set(entrypoint_target_type ${ENTRYPOINT_OBJ_VENDOR_TARGET_TYPE})
  endif()
  list(FIND TARGET_ENTRYPOINT_NAME_LIST ${ADD_ENTRYPOINT_OBJ_NAME} entrypoint_name_index)
  if(${entrypoint_name_index} EQUAL -1)
    add_custom_target(${fq_target_name})
    set_target_properties(
      ${fq_target_name}
      PROPERTIES
        "ENTRYPOINT_NAME" ${ADD_ENTRYPOINT_OBJ_NAME}
        "TARGET_TYPE" ${entrypoint_target_type}
        "OBJECT_FILE" ""
        "OBJECT_FILE_RAW" ""
        "DEPS" ""
        "SKIPPED" "YES"
    )
    if(LIBC_CMAKE_VERBOSE_LOGGING)
      message(STATUS "Skipping libc entrypoint ${fq_target_name}.")
    endif()
    return()
  endif()

  set(internal_target_name ${fq_target_name}.__internal__)

  if(ADD_ENTRYPOINT_OBJ_ALIAS)
    # Alias targets help one add aliases to other entrypoint object targets.
    # One can use alias targets setup OS/machine independent entrypoint targets.
    list(LENGTH ADD_ENTRYPOINT_OBJ_DEPENDS deps_size)
    if(NOT (${deps_size} EQUAL "1"))
      message(FATAL_ERROR "An entrypoint alias should have exactly one dependency.")
    endif()
    list(GET ADD_ENTRYPOINT_OBJ_DEPENDS 0 dep_target)
    get_fq_dep_name(fq_dep_name ${dep_target})

    if(SHOW_INTERMEDIATE_OBJECTS)
      message(STATUS "Adding entrypoint object ${fq_target_name} as an alias of"
              " ${fq_dep_name}")
    endif()

    if(NOT TARGET ${fq_dep_name})
      message(WARNING "Aliasee ${fq_dep_name} for entrypoint alias ${target_name} missing; "
                      "Target ${target_name} will be ignored.")
      return()
    endif()

    get_target_property(obj_type ${fq_dep_name} "TARGET_TYPE")
    if((NOT obj_type) OR (NOT (${obj_type} STREQUAL ${ENTRYPOINT_OBJ_TARGET_TYPE} OR
                               ${obj_type} STREQUAL ${ENTRYPOINT_OBJ_VENDOR_TARGET_TYPE})))
      message(FATAL_ERROR "The aliasee of an entrypoint alias should be an entrypoint.")
    endif()

    get_target_property(object_file ${fq_dep_name} "OBJECT_FILE")
    get_target_property(object_file_raw ${fq_dep_name} "OBJECT_FILE_RAW")
    add_library(
      ${internal_target_name}
      EXCLUDE_FROM_ALL
      OBJECT
      ${object_file_raw}
    )
    add_dependencies(${internal_target_name} ${fq_dep_name})
    add_library(
      ${fq_target_name}
      EXCLUDE_FROM_ALL
      OBJECT
      ${object_file}
    )
    add_dependencies(${fq_target_name} ${fq_dep_name} ${internal_target_name})
    set_target_properties(
      ${fq_target_name}
      PROPERTIES
        ENTRYPOINT_NAME ${ADD_ENTRYPOINT_OBJ_NAME}
        TARGET_TYPE ${entrypoint_target_type}
        IS_ALIAS "YES"
        OBJECT_FILE ""
        OBJECT_FILE_RAW ""
        DEPS "${fq_dep_name}"
        FLAGS "${ADD_ENTRYPOINT_OBJ_FLAGS}"
    )
    return()
  endif()

  if(NOT ADD_ENTRYPOINT_OBJ_SRCS)
    message(FATAL_ERROR "`add_entrypoint_object` rule requires SRCS to be specified.")
  endif()
  if(NOT ADD_ENTRYPOINT_OBJ_HDRS)
    message(FATAL_ERROR "`add_entrypoint_object` rule requires HDRS to be specified.")
  endif()
  if(NOT ADD_ENTRYPOINT_OBJ_CXX_STANDARD)
    set(ADD_ENTRYPOINT_OBJ_CXX_STANDARD ${CMAKE_CXX_STANDARD})
  endif()

  _get_common_compile_options(common_compile_options "${ADD_ENTRYPOINT_OBJ_FLAGS}")
  list(APPEND common_compile_options ${ADD_ENTRYPOINT_OBJ_COMPILE_OPTIONS})
  get_fq_deps_list(fq_deps_list ${ADD_ENTRYPOINT_OBJ_DEPENDS})
  set(full_deps_list ${fq_deps_list} libc.src.__support.common)

  if(SHOW_INTERMEDIATE_OBJECTS)
    message(STATUS "Adding entrypoint object ${fq_target_name}")
    if(${SHOW_INTERMEDIATE_OBJECTS} STREQUAL "DEPS")
      foreach(dep IN LISTS ADD_ENTRYPOINT_OBJ_DEPENDS)
        message(STATUS "  ${fq_target_name} depends on ${dep}")
      endforeach()
    endif()
  endif()

  # GPU builds require special handling for the objects because we want to
  # export several different targets at once, e.g. for both Nvidia and AMD.
  if(LIBC_TARGET_ARCHITECTURE_IS_GPU)
    _build_gpu_object_bundle(
      ${fq_target_name}
      SRCS ${ADD_ENTRYPOINT_OBJ_SRCS}
      HDRS ${ADD_ENTRYPOINT_OBJ_HDRS}
      COMPILE_OPTIONS ${common_compile_options} "-DLIBC_COPT_PUBLIC_PACKAGING"
      CXX_STANDARD ${ADD_ENTRYPOINT_OBJ_CXX_STANDARD}
      DEPENDS ${full_deps_list}
      FLAGS "${ADD_ENTRYPOINT_OBJ_FLAGS}"
    )
    _build_gpu_object_for_single_arch(
      ${internal_target_name}
      ${LIBC_GPU_TARGET_ARCHITECTURE}
      SRCS ${ADD_ENTRYPOINT_OBJ_SRCS}
      HDRS ${ADD_ENTRYPOINT_OBJ_HDRS}
      COMPILE_OPTIONS ${common_compile_options}
      CXX_STANDARD ${ADD_ENTRYPOINT_OBJ_CXX_STANDARD}
      DEPENDS ${full_deps_list}
      FLAGS "${ADD_ENTRYPOINT_OBJ_FLAGS}"
    )
  else()
    add_library(
      ${internal_target_name}
      # TODO: We don't need an object library for internal consumption.
      # A future change should switch this to a normal static library.
      EXCLUDE_FROM_ALL
      OBJECT
      ${ADD_ENTRYPOINT_OBJ_SRCS}
      ${ADD_ENTRYPOINT_OBJ_HDRS}
    )
    target_compile_options(${internal_target_name} BEFORE PRIVATE ${common_compile_options})
    target_include_directories(${internal_target_name} SYSTEM PRIVATE ${LIBC_INCLUDE_DIR})
    target_include_directories(${internal_target_name} PRIVATE ${LIBC_SOURCE_DIR})
    add_dependencies(${internal_target_name} ${full_deps_list})
    target_link_libraries(${internal_target_name} ${full_deps_list})

    add_library(
      ${fq_target_name}
      # We want an object library as the objects will eventually get packaged into
      # an archive (like libc.a).
      EXCLUDE_FROM_ALL
      OBJECT
      ${ADD_ENTRYPOINT_OBJ_SRCS}
      ${ADD_ENTRYPOINT_OBJ_HDRS}
    )
    target_compile_options(${fq_target_name} BEFORE PRIVATE ${common_compile_options} -DLIBC_COPT_PUBLIC_PACKAGING)
    target_include_directories(${fq_target_name} SYSTEM PRIVATE ${LIBC_INCLUDE_DIR})
    target_include_directories(${fq_target_name} PRIVATE ${LIBC_SOURCE_DIR})
    add_dependencies(${fq_target_name} ${full_deps_list})
    target_link_libraries(${fq_target_name} ${full_deps_list})
  endif()

  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      ENTRYPOINT_NAME ${ADD_ENTRYPOINT_OBJ_NAME}
      TARGET_TYPE ${ENTRYPOINT_OBJ_TARGET_TYPE}
      OBJECT_FILE "$<TARGET_OBJECTS:${fq_target_name}>"
      CXX_STANDARD ${ADD_ENTRYPOINT_OBJ_CXX_STANDARD}
      DEPS "${fq_deps_list}"
      FLAGS "${ADD_ENTRYPOINT_OBJ_FLAGS}"
  )

  if(TARGET ${internal_target_name})
    set_target_properties(
      ${internal_target_name}
      PROPERTIES
        CXX_STANDARD ${ADD_ENTRYPOINT_OBJ_CXX_STANDARD}
        FLAGS "${ADD_ENTRYPOINT_OBJ_FLAGS}"
    )
    set_target_properties(
      ${fq_target_name}
      PROPERTIES
        # TODO: We don't need to list internal object files if the internal
        # target is a normal static library.
        OBJECT_FILE_RAW "$<TARGET_OBJECTS:${internal_target_name}>"
    )
  endif()

  if(LLVM_LIBC_ENABLE_LINTING AND TARGET ${internal_target_name})
    if(NOT LLVM_LIBC_CLANG_TIDY)
      message(FATAL_ERROR "Something is wrong!  LLVM_LIBC_ENABLE_LINTING is "
              "ON but LLVM_LIBC_CLANG_TIDY is not set.")
    endif()

    # We only want a second invocation of clang-tidy to run
    # restrict-system-libc-headers if the compiler-resource-dir was set in
    # order to prevent false-positives due to a mismatch between the host
    # compiler and the compiled clang-tidy.
    if(COMPILER_RESOURCE_DIR)
      # We run restrict-system-libc-headers with --system-headers to prevent
      # transitive inclusion through compler provided headers.
      set(restrict_system_headers_check_invocation
        COMMAND ${LLVM_LIBC_CLANG_TIDY} --system-headers
        --checks="-*,llvmlibc-restrict-system-libc-headers"
        # We explicitly set the resource dir here to match the
        # resource dir of the host compiler.
        "--extra-arg=-resource-dir=${COMPILER_RESOURCE_DIR}"
        --quiet
        -p ${PROJECT_BINARY_DIR}
        ${ADD_ENTRYPOINT_OBJ_SRCS}
      )
    else()
      set(restrict_system_headers_check_invocation
        COMMAND ${CMAKE_COMMAND} -E echo "Header file check skipped")
    endif()

    add_custom_target(
      ${fq_target_name}.__lint__
      # --quiet is used to surpress warning statistics from clang-tidy like:
      #     Suppressed X warnings (X in non-user code).
      # There seems to be a bug in clang-tidy where by even with --quiet some
      # messages from clang's own diagnostics engine leak through:
      #     X warnings generated.
      # Until this is fixed upstream, we use -fno-caret-diagnostics to surpress
      # these.
      COMMAND ${LLVM_LIBC_CLANG_TIDY}
              "--extra-arg=-fno-caret-diagnostics" --quiet
              # Path to directory containing compile_commands.json
              -p ${PROJECT_BINARY_DIR}
              ${ADD_ENTRYPOINT_OBJ_SRCS}
      # See above: this might be a second invocation of clang-tidy depending on
      # the conditions above.
      ${restrict_system_headers_check_invocation}
      # We have two options for running commands, add_custom_command and
      # add_custom_target. We don't want to run the linter unless source files
      # have changed. add_custom_target explicitly runs everytime therefore we
      # use add_custom_command. This function requires an output file and since
      # linting doesn't produce a file, we create a dummy file using a
      # crossplatform touch.
      COMMENT "Linting... ${fq_target_name}"
      DEPENDS ${internal_target_name} ${ADD_ENTRYPOINT_OBJ_SRCS}
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    add_dependencies(libc-lint ${fq_target_name}.__lint__)
  endif()

endfunction(create_entrypoint_object)

function(add_entrypoint_object target_name)
  cmake_parse_arguments(
    "ADD_ENTRYPOINT_OBJ"
    "" # Optional arguments
    "NAME" # Single value arguments
    "" # Multi-value arguments
    ${ARGN}
  )

  if(NOT ADD_ENTRYPOINT_OBJ_NAME)
    set(ADD_ENTRYPOINT_OBJ_NAME ${target_name})
  endif()

  add_target_with_flags(
    ${target_name}
    NAME ${ADD_ENTRYPOINT_OBJ_NAME}
    CREATE_TARGET create_entrypoint_object
    ${ADD_ENTRYPOINT_OBJ_UNPARSED_ARGUMENTS}
  )
endfunction(add_entrypoint_object)

set(ENTRYPOINT_EXT_TARGET_TYPE "ENTRYPOINT_EXT")

# A rule for external entrypoint targets.
# Usage:
#     add_entrypoint_external(
#       <target_name>
#       DEPENDS <list of dependencies>
#     )
function(add_entrypoint_external target_name)
  cmake_parse_arguments(
    "ADD_ENTRYPOINT_EXT"
    "" # No optional arguments
    "" # No single value arguments
    "DEPENDS"  # Multi value arguments
    ${ARGN}
  )
  get_fq_target_name(${target_name} fq_target_name)
  set(entrypoint_name ${target_name})

  add_custom_target(${fq_target_name})
  set_target_properties(
    ${fq_target_name}
    PROPERTIES
      "ENTRYPOINT_NAME" ${entrypoint_name}
      "TARGET_TYPE" ${ENTRYPOINT_EXT_TARGET_TYPE}
      "DEPS" "${ADD_ENTRYPOINT_EXT_DEPENDS}"
  )

endfunction(add_entrypoint_external)

# Rule build a redirector object file.
function(add_redirector_object target_name)
  cmake_parse_arguments(
    "REDIRECTOR_OBJECT"
    "" # No optional arguments
    "SRC" # The cpp file in which the redirector is defined.
    "" # No multivalue arguments
    ${ARGN}
  )
  if(NOT REDIRECTOR_OBJECT_SRC)
    message(FATAL_ERROR "'add_redirector_object' rule requires SRC option listing one source file.")
  endif()

  add_library(
    ${target_name}
    EXCLUDE_FROM_ALL
    OBJECT
    ${REDIRECTOR_OBJECT_SRC}
  )
  target_compile_options(
    ${target_name}
    BEFORE PRIVATE -fPIC ${LIBC_COMPILE_OPTIONS_DEFAULT}
  )
endfunction(add_redirector_object)
