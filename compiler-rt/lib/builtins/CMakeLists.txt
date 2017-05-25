# This directory contains a large amount of C code which provides
# generic implementations of the core runtime library along with optimized
# architecture-specific code in various subdirectories.

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.4.3)

  project(CompilerRTBuiltins C ASM)
  set(COMPILER_RT_STANDALONE_BUILD TRUE)
  set(COMPILER_RT_BUILTINS_STANDALONE_BUILD TRUE)
  list(INSERT CMAKE_MODULE_PATH 0
    "${CMAKE_SOURCE_DIR}/../../cmake"
    "${CMAKE_SOURCE_DIR}/../../cmake/Modules")
  include(base-config-ix)
  include(CompilerRTUtils)

  load_llvm_config()
  construct_compiler_rt_default_triple()

  if(APPLE)
    include(CompilerRTDarwinUtils)
  endif()
  include(AddCompilerRT)
endif()

include(builtin-config-ix)

# TODO: Need to add a mechanism for logging errors when builtin source files are
# added to a sub-directory and not this CMakeLists file.
set(GENERIC_SOURCES
  absvdi2.c
  absvsi2.c
  absvti2.c
  adddf3.c
  addsf3.c
  addtf3.c
  addvdi3.c
  addvsi3.c
  addvti3.c
  apple_versioning.c
  ashldi3.c
  ashlti3.c
  ashrdi3.c
  ashrti3.c
  bswapdi2.c
  bswapsi2.c
  clear_cache.c
  clzdi2.c
  clzsi2.c
  clzti2.c
  cmpdi2.c
  cmpti2.c
  comparedf2.c
  comparesf2.c
  cpu_model.c
  ctzdi2.c
  ctzsi2.c
  ctzti2.c
  divdc3.c
  divdf3.c
  divdi3.c
  divmoddi4.c
  divmodsi4.c
  divsc3.c
  divsf3.c
  divsi3.c
  divtc3.c
  divti3.c
  divtf3.c
  divxc3.c
  eprintf.c
  extendsfdf2.c
  extendhfsf2.c
  ffsdi2.c
  ffssi2.c
  ffsti2.c
  fixdfdi.c
  fixdfsi.c
  fixdfti.c
  fixsfdi.c
  fixsfsi.c
  fixsfti.c
  fixunsdfdi.c
  fixunsdfsi.c
  fixunsdfti.c
  fixunssfdi.c
  fixunssfsi.c
  fixunssfti.c
  fixunsxfdi.c
  fixunsxfsi.c
  fixunsxfti.c
  fixxfdi.c
  fixxfti.c
  floatdidf.c
  floatdisf.c
  floatdixf.c
  floatsidf.c
  floatsisf.c
  floattidf.c
  floattisf.c
  floattixf.c
  floatundidf.c
  floatundisf.c
  floatundixf.c
  floatunsidf.c
  floatunsisf.c
  floatuntidf.c
  floatuntisf.c
  floatuntixf.c
  int_util.c
  lshrdi3.c
  lshrti3.c
  moddi3.c
  modsi3.c
  modti3.c
  muldc3.c
  muldf3.c
  muldi3.c
  mulodi4.c
  mulosi4.c
  muloti4.c
  mulsc3.c
  mulsf3.c
  multi3.c
  multf3.c
  mulvdi3.c
  mulvsi3.c
  mulvti3.c
  mulxc3.c
  negdf2.c
  negdi2.c
  negsf2.c
  negti2.c
  negvdi2.c
  negvsi2.c
  negvti2.c
  os_version_check.c
  paritydi2.c
  paritysi2.c
  parityti2.c
  popcountdi2.c
  popcountsi2.c
  popcountti2.c
  powidf2.c
  powisf2.c
  powitf2.c
  powixf2.c
  subdf3.c
  subsf3.c
  subvdi3.c
  subvsi3.c
  subvti3.c
  subtf3.c
  trampoline_setup.c
  truncdfhf2.c
  truncdfsf2.c
  truncsfhf2.c
  ucmpdi2.c
  ucmpti2.c
  udivdi3.c
  udivmoddi4.c
  udivmodsi4.c
  udivmodti4.c
  udivsi3.c
  udivti3.c
  umoddi3.c
  umodsi3.c
  umodti3.c)

set(GENERIC_TF_SOURCES
  comparetf2.c
  extenddftf2.c
  extendsftf2.c
  fixtfdi.c
  fixtfsi.c
  fixtfti.c
  fixunstfdi.c
  fixunstfsi.c
  fixunstfti.c
  floatditf.c
  floatsitf.c
  floattitf.c
  floatunditf.c
  floatunsitf.c
  floatuntitf.c
  multc3.c
  trunctfdf2.c
  trunctfsf2.c)

option(COMPILER_RT_EXCLUDE_ATOMIC_BUILTIN
  "Skip the atomic builtin (this may be needed if system headers are unavailable)"
  Off)

if(NOT COMPILER_RT_BAREMETAL_BUILD)
  set(GENERIC_SOURCES
    ${GENERIC_SOURCES}
    emutls.c 
    enable_execute_stack.c)
endif()

if(COMPILER_RT_HAS_ATOMIC_KEYWORD AND NOT COMPILER_RT_EXCLUDE_ATOMIC_BUILTIN)
  set(GENERIC_SOURCES
    ${GENERIC_SOURCES}
    atomic.c)
endif()

if(APPLE)
  set(GENERIC_SOURCES
    ${GENERIC_SOURCES}
    atomic_flag_clear.c
    atomic_flag_clear_explicit.c
    atomic_flag_test_and_set.c
    atomic_flag_test_and_set_explicit.c
    atomic_signal_fence.c
    atomic_thread_fence.c)
endif()

if (HAVE_UNWIND_H)
  set(GENERIC_SOURCES
      ${GENERIC_SOURCES}
      gcc_personality_v0.c)
endif ()

if (NOT MSVC)
  set(x86_64_SOURCES
      x86_64/chkstk.S
      x86_64/chkstk2.S
      x86_64/floatdidf.c
      x86_64/floatdisf.c
      x86_64/floatdixf.c
      x86_64/floatundidf.S
      x86_64/floatundisf.S
      x86_64/floatundixf.S
      ${GENERIC_SOURCES})
  set(x86_64h_SOURCES ${x86_64_SOURCES})

  if (WIN32)
    set(x86_64_SOURCES
        ${x86_64_SOURCES}
        x86_64/chkstk.S
        x86_64/chkstk2.S)
  endif()

  set(i386_SOURCES
      i386/ashldi3.S
      i386/ashrdi3.S
      i386/chkstk.S
      i386/chkstk2.S
      i386/divdi3.S
      i386/floatdidf.S
      i386/floatdisf.S
      i386/floatdixf.S
      i386/floatundidf.S
      i386/floatundisf.S
      i386/floatundixf.S
      i386/lshrdi3.S
      i386/moddi3.S
      i386/muldi3.S
      i386/udivdi3.S
      i386/umoddi3.S
      ${GENERIC_SOURCES})

  if (WIN32)
    set(i386_SOURCES
        ${i386_SOURCES}
        i386/chkstk.S
        i386/chkstk2.S)
  endif()

  set(i686_SOURCES
      ${i386_SOURCES})
else () # MSVC
  # Use C versions of functions when building on MSVC
  # MSVC's assembler takes Intel syntax, not AT&T syntax.
  # Also use only MSVC compilable builtin implementations.
  set(x86_64_SOURCES
      x86_64/floatdidf.c
      x86_64/floatdisf.c
      x86_64/floatdixf.c
      ${GENERIC_SOURCES})
  set(x86_64h_SOURCES ${x86_64_SOURCES})
  set(i386_SOURCES ${GENERIC_SOURCES})
  set(i686_SOURCES ${i386_SOURCES})
endif () # if (NOT MSVC)

set(arm_SOURCES
  arm/bswapdi2.S
  arm/bswapsi2.S
  arm/clzdi2.S
  arm/clzsi2.S
  arm/comparesf2.S
  arm/divmodsi4.S
  arm/divsi3.S
  arm/modsi3.S
  arm/sync_fetch_and_add_4.S
  arm/sync_fetch_and_add_8.S
  arm/sync_fetch_and_and_4.S
  arm/sync_fetch_and_and_8.S
  arm/sync_fetch_and_max_4.S
  arm/sync_fetch_and_max_8.S
  arm/sync_fetch_and_min_4.S
  arm/sync_fetch_and_min_8.S
  arm/sync_fetch_and_nand_4.S
  arm/sync_fetch_and_nand_8.S
  arm/sync_fetch_and_or_4.S
  arm/sync_fetch_and_or_8.S
  arm/sync_fetch_and_sub_4.S
  arm/sync_fetch_and_sub_8.S
  arm/sync_fetch_and_umax_4.S
  arm/sync_fetch_and_umax_8.S
  arm/sync_fetch_and_umin_4.S
  arm/sync_fetch_and_umin_8.S
  arm/sync_fetch_and_xor_4.S
  arm/sync_fetch_and_xor_8.S
  arm/udivmodsi4.S
  arm/udivsi3.S
  arm/umodsi3.S
  ${GENERIC_SOURCES})

set(thumb1_SOURCES
  arm/divsi3.S
  arm/udivsi3.S
  arm/comparesf2.S
  arm/addsf3.S
  ${GENERIC_SOURCES})

set(arm_EABI_SOURCES
  arm/aeabi_cdcmp.S
  arm/aeabi_cdcmpeq_check_nan.c
  arm/aeabi_cfcmp.S
  arm/aeabi_cfcmpeq_check_nan.c
  arm/aeabi_dcmp.S
  arm/aeabi_div0.c
  arm/aeabi_drsub.c
  arm/aeabi_fcmp.S
  arm/aeabi_frsub.c
  arm/aeabi_idivmod.S
  arm/aeabi_ldivmod.S
  arm/aeabi_memcmp.S
  arm/aeabi_memcpy.S
  arm/aeabi_memmove.S
  arm/aeabi_memset.S
  arm/aeabi_uidivmod.S
  arm/aeabi_uldivmod.S)

set(arm_Thumb1_JT_SOURCES
  arm/switch16.S
  arm/switch32.S
  arm/switch8.S
  arm/switchu8.S)
set(arm_Thumb1_SjLj_EH_SOURCES
  arm/restore_vfp_d8_d15_regs.S
  arm/save_vfp_d8_d15_regs.S)
set(arm_Thumb1_VFPv2_SOURCES
  arm/adddf3vfp.S
  arm/addsf3vfp.S
  arm/divdf3vfp.S
  arm/divsf3vfp.S
  arm/eqdf2vfp.S
  arm/eqsf2vfp.S
  arm/extendsfdf2vfp.S
  arm/fixdfsivfp.S
  arm/fixsfsivfp.S
  arm/fixunsdfsivfp.S
  arm/fixunssfsivfp.S
  arm/floatsidfvfp.S
  arm/floatsisfvfp.S
  arm/floatunssidfvfp.S
  arm/floatunssisfvfp.S
  arm/gedf2vfp.S
  arm/gesf2vfp.S
  arm/gtdf2vfp.S
  arm/gtsf2vfp.S
  arm/ledf2vfp.S
  arm/lesf2vfp.S
  arm/ltdf2vfp.S
  arm/ltsf2vfp.S
  arm/muldf3vfp.S
  arm/mulsf3vfp.S
  arm/nedf2vfp.S
  arm/negdf2vfp.S
  arm/negsf2vfp.S
  arm/nesf2vfp.S
  arm/subdf3vfp.S
  arm/subsf3vfp.S
  arm/truncdfsf2vfp.S
  arm/unorddf2vfp.S
  arm/unordsf2vfp.S)
set(arm_Thumb1_icache_SOURCES
  arm/sync_synchronize.S)
set(arm_Thumb1_SOURCES
  ${arm_Thumb1_JT_SOURCES}
  ${arm_Thumb1_SjLj_EH_SOURCES}
  ${arm_Thumb1_VFPv2_SOURCES}
  ${arm_Thumb1_icache_SOURCES})

if(MINGW)
  set(arm_SOURCES
      arm/aeabi_idivmod.S
      arm/aeabi_ldivmod.S
      arm/aeabi_uidivmod.S
      arm/aeabi_uldivmod.S
      divmoddi4.c
      divmodsi4.c
      divdi3.c
      divsi3.c
      fixdfdi.c
      fixsfdi.c
      fixunsdfdi.c
      fixunssfdi.c
      floatdidf.c
      floatdisf.c
      floatundidf.c
      floatundisf.c
      mingw_fixfloat.c
      moddi3.c
      udivmoddi4.c
      udivmodsi4.c
      udivsi3.c
      umoddi3.c
      emutls.c)
elseif(NOT WIN32)
  # TODO the EABI sources should only be added to EABI targets
  set(arm_SOURCES
    ${arm_SOURCES}
    ${arm_EABI_SOURCES}
    ${arm_Thumb1_SOURCES})

  set(thumb1_SOURCES
    ${thumb1_SOURCES}
    ${arm_EABI_SOURCES})
endif()

set(aarch64_SOURCES
  ${GENERIC_TF_SOURCES}
  ${GENERIC_SOURCES})

set(armhf_SOURCES ${arm_SOURCES})
set(armv7_SOURCES ${arm_SOURCES})
set(armv7s_SOURCES ${arm_SOURCES})
set(armv7k_SOURCES ${arm_SOURCES})
set(arm64_SOURCES ${aarch64_SOURCES})

# macho_embedded archs
set(armv6m_SOURCES ${thumb1_SOURCES})
set(armv7m_SOURCES ${arm_SOURCES})
set(armv7em_SOURCES ${arm_SOURCES})

set(mips_SOURCES ${GENERIC_SOURCES})
set(mipsel_SOURCES ${mips_SOURCES})
set(mips64_SOURCES ${GENERIC_TF_SOURCES}
                   ${mips_SOURCES})
set(mips64el_SOURCES ${GENERIC_TF_SOURCES}
                     ${mips_SOURCES})

set(wasm32_SOURCES ${GENERIC_SOURCES})
set(wasm64_SOURCES ${GENERIC_SOURCES})

add_custom_target(builtins)
set_target_properties(builtins PROPERTIES FOLDER "Compiler-RT Misc")

if (APPLE)
  add_subdirectory(Darwin-excludes)
  add_subdirectory(macho_embedded)
  darwin_add_builtin_libraries(${BUILTIN_SUPPORTED_OS})
else ()
  set(BUILTIN_CFLAGS "")

  append_list_if(COMPILER_RT_HAS_STD_C11_FLAG -std=c11 BUILTIN_CFLAGS)

  # These flags would normally be added to CMAKE_C_FLAGS by the llvm
  # cmake step. Add them manually if this is a standalone build.
  if(COMPILER_RT_STANDALONE_BUILD)
    append_list_if(COMPILER_RT_HAS_FPIC_FLAG -fPIC BUILTIN_CFLAGS)
    append_list_if(COMPILER_RT_HAS_FNO_BUILTIN_FLAG -fno-builtin BUILTIN_CFLAGS)
    append_list_if(COMPILER_RT_HAS_VISIBILITY_HIDDEN_FLAG -fvisibility=hidden BUILTIN_CFLAGS)
    if(NOT COMPILER_RT_DEBUG)
      append_list_if(COMPILER_RT_HAS_OMIT_FRAME_POINTER_FLAG -fomit-frame-pointer BUILTIN_CFLAGS)
    endif()
  endif()

  set(BUILTIN_DEFS "")

  append_list_if(COMPILER_RT_HAS_VISIBILITY_HIDDEN_FLAG VISIBILITY_HIDDEN BUILTIN_DEFS)

  foreach (arch ${BUILTIN_SUPPORTED_ARCH})
    if (CAN_TARGET_${arch})
      # NOTE: some architectures (e.g. i386) have multiple names.  Ensure that
      # we catch them all.
      set(_arch ${arch})
      if("${arch}" STREQUAL "i686")
        set(_arch "i386|i686")
      endif()

      # Filter out generic versions of routines that are re-implemented in
      # architecture specific manner.  This prevents multiple definitions of the
      # same symbols, making the symbol selection non-deterministic.
      foreach (_file ${${arch}_SOURCES})
        if (${_file} MATCHES ${_arch}/*)
          get_filename_component(_name ${_file} NAME)
          string(REPLACE ".S" ".c" _cname "${_name}")
          list(REMOVE_ITEM ${arch}_SOURCES ${_cname})
        endif ()
      endforeach ()

      # Needed for clear_cache on debug mode, due to r7's usage in inline asm.
      # Release mode already sets it via -O2/3, Debug mode doesn't.
      if (${arch} STREQUAL "armhf")
        list(APPEND BUILTIN_CFLAGS -fomit-frame-pointer -DCOMPILER_RT_ARMHF_TARGET)
      endif()

      add_compiler_rt_runtime(clang_rt.builtins
                              STATIC
                              ARCHS ${arch}
                              SOURCES ${${arch}_SOURCES}
                              DEFS ${BUILTIN_DEFS}
                              CFLAGS ${BUILTIN_CFLAGS}
                              PARENT_TARGET builtins)
    endif ()
  endforeach ()
endif ()

add_dependencies(compiler-rt builtins)
