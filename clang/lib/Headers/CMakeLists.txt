set(files
  adxintrin.h
  altivec.h
  ammintrin.h
  arm_acle.h
  armintr.h
  arm64intr.h
  avx2intrin.h
  avx512bwintrin.h
  avx512cdintrin.h
  avx512vpopcntdqintrin.h
  avx512dqintrin.h
  avx512erintrin.h
  avx512fintrin.h
  avx512ifmaintrin.h
  avx512ifmavlintrin.h
  avx512pfintrin.h
  avx512vbmiintrin.h
  avx512vbmivlintrin.h
  avx512vlbwintrin.h
  avx512vlcdintrin.h
  avx512vldqintrin.h
  avx512vlintrin.h
  avxintrin.h
  bmi2intrin.h
  bmiintrin.h
  __clang_cuda_builtin_vars.h
  __clang_cuda_cmath.h
  __clang_cuda_complex_builtins.h
  __clang_cuda_intrinsics.h
  __clang_cuda_math_forward_declares.h
  __clang_cuda_runtime_wrapper.h
  cetintrin.h
  clzerointrin.h
  cpuid.h
  clflushoptintrin.h
  clwbintrin.h
  emmintrin.h
  f16cintrin.h
  float.h
  fma4intrin.h
  fmaintrin.h
  fxsrintrin.h
  htmintrin.h
  htmxlintrin.h
  ia32intrin.h
  immintrin.h
  intrin.h
  inttypes.h
  iso646.h
  limits.h
  lwpintrin.h
  lzcntintrin.h
  mm3dnow.h
  mmintrin.h
  mm_malloc.h
  module.modulemap
  msa.h
  mwaitxintrin.h
  nmmintrin.h
  opencl-c.h
  pkuintrin.h
  pmmintrin.h
  popcntintrin.h
  prfchwintrin.h
  rdseedintrin.h
  rtmintrin.h
  s390intrin.h
  shaintrin.h
  smmintrin.h
  stdalign.h
  stdarg.h
  stdatomic.h
  stdbool.h
  stddef.h
  __stddef_max_align_t.h
  stdint.h
  stdnoreturn.h
  tbmintrin.h
  tgmath.h
  tmmintrin.h
  unwind.h
  vadefs.h
  varargs.h
  vecintrin.h
  wmmintrin.h
  __wmmintrin_aes.h
  __wmmintrin_pclmul.h
  x86intrin.h
  xmmintrin.h
  xopintrin.h
  xsavecintrin.h
  xsaveintrin.h
  xsaveoptintrin.h
  xsavesintrin.h
  xtestintrin.h
  )

set(cuda_wrapper_files
  cuda_wrappers/algorithm
  cuda_wrappers/complex
  cuda_wrappers/new
)

set(output_dir ${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION}/include)

# Generate arm_neon.h
clang_tablegen(arm_neon.h -gen-arm-neon
  SOURCE ${CLANG_SOURCE_DIR}/include/clang/Basic/arm_neon.td)

set(out_files)
foreach( f ${files} ${cuda_wrapper_files} )
  set( src ${CMAKE_CURRENT_SOURCE_DIR}/${f} )
  set( dst ${output_dir}/${f} )
  add_custom_command(OUTPUT ${dst}
    DEPENDS ${src}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${src} ${dst}
    COMMENT "Copying clang's ${f}...")
  list(APPEND out_files ${dst})
endforeach( f )

add_custom_command(OUTPUT ${output_dir}/arm_neon.h 
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/arm_neon.h
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_BINARY_DIR}/arm_neon.h ${output_dir}/arm_neon.h
  COMMENT "Copying clang's arm_neon.h...")
list(APPEND out_files ${output_dir}/arm_neon.h)

add_custom_target(clang-headers ALL DEPENDS ${out_files})
set_target_properties(clang-headers PROPERTIES FOLDER "Misc")

install(
  FILES ${files} ${CMAKE_CURRENT_BINARY_DIR}/arm_neon.h
  COMPONENT clang-headers
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION}/include)

install(
  FILES ${cuda_wrapper_files}
  COMPONENT clang-headers
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION}/include/cuda_wrappers)

if (NOT CMAKE_CONFIGURATION_TYPES) # don't add this for IDE's.
  add_custom_target(install-clang-headers
    DEPENDS clang-headers
    COMMAND "${CMAKE_COMMAND}"
            -DCMAKE_INSTALL_COMPONENT=clang-headers
            -P "${CMAKE_BINARY_DIR}/cmake_install.cmake")
endif()
