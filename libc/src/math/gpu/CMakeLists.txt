# Math functions not yet available in the libc project, or those not yet tuned
# for GPU workloads are provided as wrappers over vendor libraries. If we find
# them ahead of time we will import them statically. Otherwise, we will keep
# them as external references and expect them to be resolved by the user when
# they compile. In the future,we will use implementations from the 'libc'
# project and not provide these wrappers.
add_subdirectory(vendor)

# For the GPU we want to be able to optionally depend on the vendor libraries
# until we have a suitable replacement inside `libc`.
# TODO: We should have an option to enable or disable these on a per-function
# basis.
option(LIBC_GPU_VENDOR_MATH "Use vendor wrappers for GPU math" ON)
function(add_math_entrypoint_gpu_object name)
  get_fq_target_name("vendor.${name}" fq_vendor_specific_target_name)
  if(TARGET ${fq_vendor_specific_target_name} AND ${LIBC_GPU_VENDOR_MATH})
    return()
  endif()

  add_entrypoint_object(
    ${name}
    ${ARGN}
  )
endfunction()

add_math_entrypoint_gpu_object(
  round
  SRCS
    round.cpp
  HDRS
    ../round.h
  COMPILE_OPTIONS
    -O2
)
