# core_files list contains the headers shared by all platforms.
# Please consider adding new platform specific headers
# to platform specific lists below.
set(core_files
  builtins.h
  float.h
  inttypes.h
  iso646.h
  limits.h
  module.modulemap
  stdalign.h
  stdarg.h
  stdatomic.h
  stdbool.h
  stddef.h
  __stddef_max_align_t.h
  stdint.h
  stdnoreturn.h
  tgmath.h
  unwind.h
  varargs.h
  )

set(arm_common_files
  # Headers shared by Arm and AArch64
  arm_acle.h
  )

set(arm_only_files
  arm_cmse.h
  armintr.h
  )

set(aarch64_only_files
  arm64intr.h
  )

set(cuda_files
  __clang_cuda_builtin_vars.h
  __clang_cuda_math.h
  __clang_cuda_cmath.h
  __clang_cuda_complex_builtins.h
  __clang_cuda_device_functions.h
  __clang_cuda_intrinsics.h
  __clang_cuda_texture_intrinsics.h
  __clang_cuda_libdevice_declares.h
  __clang_cuda_math_forward_declares.h
  __clang_cuda_runtime_wrapper.h
  )

set(hexagon_files
  hexagon_circ_brev_intrinsics.h
  hexagon_protos.h
  hexagon_types.h
  hvx_hexagon_protos.h
  )

set(hip_files
  __clang_hip_libdevice_declares.h
  __clang_hip_cmath.h
  __clang_hip_math.h
  __clang_hip_runtime_wrapper.h
  )

set(mips_msa_files
  msa.h
  )

set(opencl_files
  opencl-c.h
  opencl-c-base.h
  )

set(ppc_files
  altivec.h
  htmintrin.h
  htmxlintrin.h
  )

set(systemz_files
  s390intrin.h
  vecintrin.h
  )

set(ve_files
  velintrin.h
  velintrin_gen.h
  velintrin_approx.h
  )

set(webassembly_files
  wasm_simd128.h
  )

set(x86_files
# Intrinsics
  adxintrin.h
  ammintrin.h
  amxintrin.h
  avx2intrin.h
  avx512bf16intrin.h
  avx512bitalgintrin.h
  avx512bwintrin.h
  avx512cdintrin.h
  avx512dqintrin.h
  avx512erintrin.h
  avx512fintrin.h
  avx512fp16intrin.h
  avx512ifmaintrin.h
  avx512ifmavlintrin.h
  avx512pfintrin.h
  avx512vbmi2intrin.h
  avx512vbmiintrin.h
  avx512vbmivlintrin.h
  avx512vlbf16intrin.h
  avx512vlbitalgintrin.h
  avx512vlbwintrin.h
  avx512vlcdintrin.h
  avx512vldqintrin.h
  avx512vlfp16intrin.h
  avx512vlintrin.h
  avx512vlvbmi2intrin.h
  avx512vlvnniintrin.h
  avx512vlvp2intersectintrin.h
  avx512vnniintrin.h
  avx512vp2intersectintrin.h
  avx512vpopcntdqintrin.h
  avx512vpopcntdqvlintrin.h
  avxintrin.h
  avxvnniintrin.h
  bmi2intrin.h
  bmiintrin.h
  cetintrin.h
  cldemoteintrin.h
  clflushoptintrin.h
  clwbintrin.h
  clzerointrin.h
  crc32intrin.h
  emmintrin.h
  enqcmdintrin.h
  f16cintrin.h
  fma4intrin.h
  fmaintrin.h
  fxsrintrin.h
  gfniintrin.h
  hresetintrin.h
  ia32intrin.h
  immintrin.h
  invpcidintrin.h
  keylockerintrin.h
  lwpintrin.h
  lzcntintrin.h
  mm3dnow.h
  mmintrin.h
  movdirintrin.h
  mwaitxintrin.h
  nmmintrin.h
  pconfigintrin.h
  pkuintrin.h
  pmmintrin.h
  popcntintrin.h
  prfchwintrin.h
  ptwriteintrin.h
  rdseedintrin.h
  rtmintrin.h
  serializeintrin.h
  sgxintrin.h
  shaintrin.h
  smmintrin.h
  tbmintrin.h
  tmmintrin.h
  tsxldtrkintrin.h
  uintrintrin.h
  vaesintrin.h
  vpclmulqdqintrin.h
  waitpkgintrin.h
  wbnoinvdintrin.h
  __wmmintrin_aes.h
  wmmintrin.h
  __wmmintrin_pclmul.h
  x86gprintrin.h
  x86intrin.h
  xmmintrin.h
  xopintrin.h
  xsavecintrin.h
  xsaveintrin.h
  xsaveoptintrin.h
  xsavesintrin.h
  xtestintrin.h
# others
  cet.h
  cpuid.h
  )

set(windows_only_files
  intrin.h
  vadefs.h
)

set(utility_files
  mm_malloc.h
)

set(files
  ${core_files}
  ${arm_common_files}
  ${arm_only_files}
  ${aarch64_only_files}
  ${cuda_files}
  ${hexagon_files}
  ${hip_files}
  ${mips_msa_files}
  ${opencl_files}
  ${ppc_files}
  ${systemz_files}
  ${ve_files}
  ${x86_files}
  ${webassembly_files}
  ${windows_only_files}
  ${utility_files}
  )

set(cuda_wrapper_files
  cuda_wrappers/algorithm
  cuda_wrappers/complex
  cuda_wrappers/new
)

set(ppc_wrapper_files
  ppc_wrappers/mmintrin.h
  ppc_wrappers/xmmintrin.h
  ppc_wrappers/mm_malloc.h
  ppc_wrappers/emmintrin.h
  ppc_wrappers/pmmintrin.h
  ppc_wrappers/tmmintrin.h
  ppc_wrappers/smmintrin.h
  ppc_wrappers/bmiintrin.h
  ppc_wrappers/bmi2intrin.h
  ppc_wrappers/immintrin.h
  ppc_wrappers/x86intrin.h
  ppc_wrappers/x86gprintrin.h
)

set(openmp_wrapper_files
  openmp_wrappers/math.h
  openmp_wrappers/cmath
  openmp_wrappers/complex.h
  openmp_wrappers/complex
  openmp_wrappers/__clang_openmp_device_functions.h
  openmp_wrappers/complex_cmath.h
  openmp_wrappers/new
)

set(output_dir ${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION}/include)
set(out_files)
set(generated_files)

set(arm_common_generated_files)
set(arm_only_generated_files)
set(aarch64_only_generated_files)
set(riscv_generated_files)

function(copy_header_to_output_dir src_dir file)
  set(src ${src_dir}/${file})
  set(dst ${output_dir}/${file})
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying clang's ${file}...")
  list(APPEND out_files ${dst})
  set(out_files ${out_files} PARENT_SCOPE)
endfunction(copy_header_to_output_dir)

function(clang_generate_header td_option td_file out_file)
  clang_tablegen(${out_file} ${td_option}
  -I ${CLANG_SOURCE_DIR}/include/clang/Basic/
  SOURCE ${CLANG_SOURCE_DIR}/include/clang/Basic/${td_file})

  copy_header_to_output_dir(${CMAKE_CURRENT_BINARY_DIR} ${out_file})
  set(out_files ${out_files} PARENT_SCOPE)
  list(APPEND generated_files "${CMAKE_CURRENT_BINARY_DIR}/${out_file}")
  set(generated_files ${generated_files} PARENT_SCOPE)
endfunction(clang_generate_header)


# Copy header files from the source directory to the build directory
foreach( f ${files} ${cuda_wrapper_files} ${ppc_wrapper_files} ${openmp_wrapper_files})
  copy_header_to_output_dir(${CMAKE_CURRENT_SOURCE_DIR} ${f})
endforeach( f )

# Generate header files and copy them to the build directory
if(ARM IN_LIST LLVM_TARGETS_TO_BUILD OR AArch64 IN_LIST LLVM_TARGETS_TO_BUILD)
  # Generate arm_neon.h
  clang_generate_header(-gen-arm-neon arm_neon.td arm_neon.h)
  # Generate arm_fp16.h
  clang_generate_header(-gen-arm-fp16 arm_fp16.td arm_fp16.h)
  # Generate arm_sve.h
  clang_generate_header(-gen-arm-sve-header arm_sve.td arm_sve.h)
  # Generate arm_bf16.h
  clang_generate_header(-gen-arm-bf16 arm_bf16.td arm_bf16.h)
  # Generate arm_mve.h
  clang_generate_header(-gen-arm-mve-header arm_mve.td arm_mve.h)
  # Generate arm_cde.h
  clang_generate_header(-gen-arm-cde-header arm_cde.td arm_cde.h)
  # Copy arm_neon_sve_bridge.h
  copy_header_to_output_dir(${CMAKE_CURRENT_SOURCE_DIR}
    arm_neon_sve_bridge.h
  )

  # Add headers to target specific lists
  list(APPEND arm_common_generated_files
    "${CMAKE_CURRENT_BINARY_DIR}/arm_neon.h"
    "${CMAKE_CURRENT_BINARY_DIR}/arm_fp16.h"
    )

  list(APPEND arm_only_generated_files
    "${CMAKE_CURRENT_BINARY_DIR}/arm_mve.h"
    "${CMAKE_CURRENT_BINARY_DIR}/arm_cde.h"
    )

  list(APPEND aarch64_only_generated_files
    "${CMAKE_CURRENT_BINARY_DIR}/arm_sve.h"
    "${CMAKE_CURRENT_BINARY_DIR}/arm_bf16.h"
    "${output_dir}/arm_neon_sve_bridge.h"
    )
endif()
if(RISCV IN_LIST LLVM_TARGETS_TO_BUILD)
  # Generate riscv_vector.h
  clang_generate_header(-gen-riscv-vector-header riscv_vector.td riscv_vector.h)
  list(APPEND riscv_generated_files
    "${CMAKE_CURRENT_BINARY_DIR}/riscv_vector.h"
    )
endif()


# Check if the generated headers are included in a target specific lists
# Currently, all generated headers are target specific.
set(all_target_specific_generated_files
  ${arm_common_generated_files}
  ${arm_only_generated_files}
  ${aarch64_only_generated_files}
  ${riscv_generated_files})
foreach( f ${generated_files} )
  if (NOT ${f} IN_LIST all_target_specific_generated_files)
    message(WARNING "${f} is a generated header but it is not included in any "
                    "target specific header lists! The resource-headers "
                    "distribtion target will be incorrect!")
  endif()
endforeach( f )

function(add_header_target target_name file_list)
  add_custom_target(${target_name} DEPENDS ${file_list})
  set_target_properties(${target_name} PROPERTIES
    FOLDER "Misc"
    RUNTIME_OUTPUT_DIRECTORY "${output_dir}")
endfunction()

# The catch-all clang-resource-headers target
add_custom_target("clang-resource-headers" ALL DEPENDS ${out_files})
set_target_properties("clang-resource-headers" PROPERTIES
  FOLDER "Misc"
  RUNTIME_OUTPUT_DIRECTORY "${output_dir}")
add_dependencies("clang-resource-headers"
                 "core-resource-headers"
                 "arm-common-headers"
                 "arm-resource-headers"
                 "aarch64-resource-headers"
                 "cuda-resource-headers"
                 "hexagon-resource-headers"
                 "hip-resource-headers"
                 "mips-resource-headers"
                 "ppc-resource-headers"
                 "riscv-resource-headers"
                 "systemz-resource-headers"
                 "ve-resource-headers"
                 "webassembly-resource-headers"
                 "x86-resource-headers"
                 "opencl-resource-headers"
                 "openmp-resource-headers"
                 "windows-resource-headers"
                 "utility-resource-headers")

# Core/common headers
add_header_target("core-resource-headers" ${core_files})
add_header_target("arm-common-headers" "${arm_common_files};${arm_common_generated_files}")

# Architecture/platform specific targets
add_header_target("arm-resource-headers" "${arm_only_files};${arm_only_generated_files}")
add_header_target("aarch64-resource-headers" "${aarch64_only_files};${aarch64_only_generated_files}")
add_header_target("cuda-resource-headers" "${cuda_files};${cuda_wrapper_files}")
add_header_target("hexagon-resource-headers" "${hexagon_files}")
add_header_target("hip-resource-headers" "${hip_files}")
add_header_target("mips-resource-headers" "${mips_msa_files}")
add_header_target("ppc-resource-headers" "${ppc_files};${ppc_wrapper_files}")
add_header_target("riscv-resource-headers" "${riscv_generated_files}")
add_header_target("systemz-resource-headers" "${systemz_files}")
add_header_target("ve-resource-headers" "${ve_files}")
add_header_target("webassembly-resource-headers" "${webassembly_files}")
add_header_target("x86-resource-headers" "${x86_files}")

# Other header groupings
add_header_target("opencl-resource-headers" ${opencl_files})
add_header_target("openmp-resource-headers" ${openmp_wrapper_files})
add_header_target("windows-resource-headers" ${windows_only_files})
add_header_target("utility-resource-headers" ${utility_files})

set(header_install_dir lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION}/include)

#############################################################
# Install rules for the catch-all clang-resource-headers target
install(
  FILES ${files} ${generated_files}
  DESTINATION ${header_install_dir}
  COMPONENT clang-resource-headers)

install(
  FILES ${cuda_wrapper_files}
  DESTINATION ${header_install_dir}/cuda_wrappers
  COMPONENT clang-resource-headers)

install(
  FILES ${ppc_wrapper_files}
  DESTINATION ${header_install_dir}/ppc_wrappers
  COMPONENT clang-resource-headers)

install(
  FILES ${openmp_wrapper_files}
  DESTINATION ${header_install_dir}/openmp_wrappers
  COMPONENT clang-resource-headers)

#############################################################
# Install rules for separate header lists
install(
  FILES ${core_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT core-resource-headers)

install(
  FILES ${arm_common_files} ${arm_common_generated_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT arm-common-resource-headers)

install(
  FILES ${arm_only_files} ${arm_only_generated_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT arm-resource-headers)

install(
  FILES ${aarch64_only_files} ${aarch64_only_generated_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT aarch64-resource-headers)

install(
  FILES ${cuda_wrapper_files}
  DESTINATION ${header_install_dir}/cuda_wrappers
  EXCLUDE_FROM_ALL
  COMPONENT cuda-resource-headers)

install(
  FILES ${cuda_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT cuda-resource-headers)

install(
  FILES ${hexagon_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT hexagon-resource-headers)

install(
  FILES ${hip_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT hip-resource-headers)

install(
  FILES ${mips_msa_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT mips-resource-headers)

install(
  FILES ${ppc_wrapper_files}
  DESTINATION ${header_install_dir}/ppc_wrappers
  EXCLUDE_FROM_ALL
  COMPONENT ppc-resource-headers)

install(
  FILES ${ppc_files} ${utility_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT ppc-resource-headers)

install(
  FILES ${riscv_generated_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT riscv-resource-headers)

install(
  FILES ${systemz_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT systemz-resource-headers)

install(
  FILES ${ve_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT ve-resource-headers)

install(
  FILES ${webassembly_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT webassembly-resource-headers)

install(
  FILES ${x86_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT x86-resource-headers)

install(
  FILES ${opencl_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT opencl-resource-headers)

install(
  FILES ${openmp_wrapper_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT openmp-resource-headers)

install(
  FILES ${utility_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT utility-resource-headers)

install(
  FILES ${windows_files}
  DESTINATION ${header_install_dir}
  EXCLUDE_FROM_ALL
  COMPONENT windows-resource-headers)
#############################################################

if (NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-clang-resource-headers
                           DEPENDS clang-resource-headers
                           COMPONENT clang-resource-headers)

  add_llvm_install_targets(install-core-resource-headers
                           DEPENDS core-resource-headers
                           COMPONENT core-resource-headers)
  add_llvm_install_targets(install-arm-common-resource-headers
                           DEPENDS arm-common-resource-headers
                           COMPONENT arm-common-resource-headers)
  add_llvm_install_targets(install-arm-resource-headers
                           DEPENDS arm-resource-headers
                           COMPONENT arm-resource-headers)
  add_llvm_install_targets(install-aarch64-resource-headers
                           DEPENDS aarch64-resource-headers
                           COMPONENT aarch64-resource-headers)
  add_llvm_install_targets(install-cuda-resource-headers
                           DEPENDS cuda-resource-headers
                           COMPONENT cuda-resource-headers)
  add_llvm_install_targets(install-hexagon-resource-headers
                           DEPENDS hexagon-resource-headers
                           COMPONENT hexagon-resource-headers)
  add_llvm_install_targets(install-hip-resource-headers
                           DEPENDS hip-resource-headers
                           COMPONENT hip-resource-headers)
  add_llvm_install_targets(install-mips-resource-headers
                           DEPENDS mips-resource-headers
                           COMPONENT mips-resource-headers)
  add_llvm_install_targets(install-ppc-resource-headers
                           DEPENDS ppc-resource-headers
                           COMPONENT ppc-resource-headers)
  add_llvm_install_targets(install-riscv-resource-headers
                           DEPENDS riscv-resource-headers
                           COMPONENT riscv-resource-headers)
  add_llvm_install_targets(install-systemz-resource-headers
                           DEPENDS systemz-resource-headers
                           COMPONENT systemz-resource-headers)
  add_llvm_install_targets(install-ve-resource-headers
                           DEPENDS ve-resource-headers
                           COMPONENT ve-resource-headers)
  add_llvm_install_targets(install-x86-resource-headers
                           DEPENDS x86-resource-headers
                           COMPONENT x86-resource-headers)
  add_llvm_install_targets(install-webassembly-resource-headers
                           DEPENDS webassembly-resource-headers
                           COMPONENT webassembly-resource-headers)

  add_llvm_install_targets(install-opencl-resource-headers
                           DEPENDS opencl-resource-headers
                           COMPONENT opencl-resource-headers)
  add_llvm_install_targets(install-openmp-resource-headers
                           DEPENDS openmp-resource-headers
                           COMPONENT openmp-resource-headers)
  add_llvm_install_targets(install-windows-resource-headers
                           DEPENDS window-resource-headers
                           COMPONENT windows-resource-headers)
  add_llvm_install_targets(install-utility-resource-headers
                           DEPENDS utility-resource-headers
                           COMPONENT utility-resource-headers)
endif()
