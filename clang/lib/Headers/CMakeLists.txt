set(files
  adxintrin.h
  altivec.h
  ammintrin.h
  arm_acle.h
  avx2intrin.h
  avx512bwintrin.h
  avx512cdintrin.h
  avx512erintrin.h
  avx512fintrin.h
  avx512vlbwintrin.h
  avx512vlintrin.h
  avx512dqintrin.h
  avx512vldqintrin.h
  avxintrin.h
  bmi2intrin.h
  bmiintrin.h
  cpuid.h
  cuda_builtin_vars.h
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
  Intrin.h
  inttypes.h
  iso646.h
  limits.h
  lzcntintrin.h
  mm3dnow.h
  mmintrin.h
  mm_malloc.h
  module.modulemap
  nmmintrin.h
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
  __wmmintrin_aes.h
  wmmintrin.h
  __wmmintrin_pclmul.h
  x86intrin.h
  xmmintrin.h
  xopintrin.h
  xsaveintrin.h
  xsaveoptintrin.h
  xsavecintrin.h
  xsavesintrin.h
  xtestintrin.h
  )

set(output_dir ${LLVM_LIBRARY_OUTPUT_INTDIR}/clang/${CLANG_VERSION}/include)

# Generate arm_neon.h
clang_tablegen(arm_neon.h -gen-arm-neon
  SOURCE ${CLANG_SOURCE_DIR}/include/clang/Basic/arm_neon.td)

set(out_files)
foreach( f ${files} )
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
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
  DESTINATION lib${LLVM_LIBDIR_SUFFIX}/clang/${CLANG_VERSION}/include)
