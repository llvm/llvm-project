#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

  if(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "x86_64|ppc64le|aarch64" AND NOT ${LIBPGMATH_WITH_GENERIC})
    add_subdirectory("acos")
    add_subdirectory("asin")
    add_subdirectory("exp")
    add_subdirectory("log")
    add_subdirectory("pow")
  endif()
  if(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "x86_64" AND NOT ${LIBPGMATH_WITH_GENERIC})
    add_subdirectory("cexp")
    add_subdirectory("cexpf")
    add_subdirectory("log10")
    add_subdirectory("log10f")
    add_subdirectory("logf")
    add_subdirectory("atan")
    add_subdirectory("atanf")
    add_subdirectory("atan2")
    add_subdirectory("atan2f")
  endif()
    add_subdirectory("powi")
    add_subdirectory("sincos")

# Get compiler definitions
get_property(FLAGS GLOBAL PROPERTY "FLAGS_L1")
get_property(DEFINITIONS GLOBAL PROPERTY "DEFINITIONS_L1")

if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Linux")
  set(DEFINITIONS "${DEFINITIONS}" _GNU_SOURCE)
endif()

set(SRCS)
if(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "x86_64" AND NOT ${LIBPGMATH_WITH_GENERIC})
  if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Linux")
    add_subdirectory("sincosf")
    add_subdirectory("tanf")

    # Specific definition, will probably need to remove when we create a single file for each def
    set_property(SOURCE dispatch.c APPEND_STRING PROPERTY COMPILE_FLAGS "-fno-builtin-c{sqrt,pow,log,exp,acos,asin,atan,cos,sin,tan,cosh,sinh,tanh}{,f} -fno-builtin-{ceil,floor}{,f}")
    set_property(SOURCE dispatch.c APPEND PROPERTY COMPILE_DEFINITIONS PGFLANG)
    set_property(SOURCE mth_xintrinsics.c APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core2 ")
    set_property(SOURCE mth_yintrinsics.c APPEND_STRING PROPERTY COMPILE_FLAGS "-march=sandybridge ")
    set_property(SOURCE mth_zintrinsics.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mtune=knl -march=knl ")
    set_property(SOURCE mth_vreturns.c APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core2 ")
    set_property(SOURCE p_dpowi.c APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core2 ")
    set_property(SOURCE p_rpowi.c APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core2 ")

    set_property(SOURCE mth_128defs.c mth_128defs_init.c mth_128defs_stats.c APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core2 ")
    set_property(SOURCE mth_128defs_init.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_INIT)
    set_property(SOURCE mth_128defs_stats.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_STATS)

    set_property(SOURCE mth_256defs.c mth_256defs_init.c mth_256defs_stats.c APPEND_STRING PROPERTY COMPILE_FLAGS "-march=sandybridge ")
    set_property(SOURCE mth_256defs_init.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_INIT)
    set_property(SOURCE mth_256defs_stats.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_STATS)

    set_property(SOURCE mth_512defs.c mth_512defs_init.c mth_512defs_stats.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mtune=knl -march=knl ")
    set_property(SOURCE mth_512defs_init.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_INIT)
    set_property(SOURCE mth_512defs_stats.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_STATS)
    set(MTH_INTRINSICS_SRCS
      mth_128defs.c
      mth_128defs_init.c
      mth_128defs_stats.c
      mth_256defs.c
      mth_256defs_init.c
      mth_256defs_stats.c
      mth_512defs.c
      mth_512defs_init.c
      mth_512defs_stats.c)
    libmath_add_object_library("${MTH_INTRINSICS_SRCS}" "${FLAGS}" "${DEFINITIONS}" "mth_intrinsics")

    set(SRCS
      mth_yintrinsics.c
      mth_zintrinsics.c
      kidnnt.c)
  elseif(${LIBPGMATH_SYSTEM_NAME} MATCHES "Darwin|Windows")
    add_subdirectory("sincosf")
    add_subdirectory("tanf")

    # Specific definition, will probably need to remove when we create a single file for each def
    set_property(SOURCE dispatch.c APPEND_STRING PROPERTY COMPILE_FLAGS "-fno-builtin-c{sqrt,pow,log,exp,acos,asin,atan,cos,sin,tan,cosh,sinh,tanh}{,f} -fno-builtin-{ceil,floor}{,f}")
    set_property(SOURCE dispatch.c APPEND PROPERTY COMPILE_DEFINITIONS PGFLANG)
    set_property(SOURCE mth_xintrinsics.c APPEND_STRING PROPERTY COMPILE_FLAGS "-msse2 ")
    set_property(SOURCE mth_yintrinsics.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mavx ")
    set_property(SOURCE mth_zintrinsics.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mavx512f -mfma ")
    set_property(SOURCE mth_vreturns.c APPEND_STRING PROPERTY COMPILE_FLAGS "-msse2 ")
    set_property(SOURCE p_dpowi.c APPEND_STRING PROPERTY COMPILE_FLAGS "-msse2 ")
    set_property(SOURCE p_rpowi.c APPEND_STRING PROPERTY COMPILE_FLAGS "-msse2 ")

    set_property(SOURCE mth_128defs.c mth_128defs_init.c mth_128defs_stats.c APPEND_STRING PROPERTY COMPILE_FLAGS "-msse2 ")
    set_property(SOURCE mth_128defs_init.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_INIT)
    set_property(SOURCE mth_128defs_stats.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_STATS)

    set_property(SOURCE mth_256defs.c mth_256defs_init.c mth_256defs_stats.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mavx ")
    set_property(SOURCE mth_256defs_init.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_INIT)
    set_property(SOURCE mth_256defs_stats.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_STATS)

    set_property(SOURCE mth_512defs.c mth_512defs_init.c mth_512defs_stats.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mavx512f ")
    set_property(SOURCE mth_512defs_init.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_INIT)
    set_property(SOURCE mth_512defs_stats.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_STATS)
    set(MTH_INTRINSICS_SRCS
      mth_128defs.c
      mth_128defs_init.c
      mth_128defs_stats.c
      mth_256defs.c
      mth_256defs_init.c
      mth_256defs_stats.c
      mth_512defs.c
      mth_512defs_init.c
      mth_512defs_stats.c)
    libmath_add_object_library("${MTH_INTRINSICS_SRCS}" "${FLAGS}" "${DEFINITIONS}" "mth_intrinsics")

    set(SRCS
      mth_yintrinsics.c
      mth_zintrinsics.c
      kidnnt.c)
  endif()
elseif(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "ppc64le" AND NOT ${LIBPGMATH_WITH_GENERIC})
  set_property(SOURCE dispatch.c APPEND_STRING PROPERTY COMPILE_FLAGS "-fno-builtin")
  set_property(SOURCE mth_xintrinsics.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mcpu=power8 ")
  set_property(SOURCE mth_128defs.c mth_128defs_init.c mth_128defs_stats.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mcpu=power8 ")
  set_property(SOURCE mth_128defs_init.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_INIT)
  set_property(SOURCE mth_128defs_stats.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_STATS)
  set(MTH_INTRINSICS_SRCS
    mth_128defs.c
    mth_128defs_init.c
    mth_128defs_stats.c)
  libmath_add_object_library("${MTH_INTRINSICS_SRCS}" "${FLAGS}" "${DEFINITIONS}" "mth_intrinsics")

  set(SRCS
    kidnnt.c
    sincos.c
    sincosf.c)
elseif(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "aarch64" AND NOT ${LIBPGMATH_WITH_GENERIC})
  if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Windows")
    set_property(SOURCE mth_128defs.c mth_128defs_init.c mth_128defs_stats.c APPEND_STRING PROPERTY COMPILE_FLAGS "--target=aarch64-windows-msvc ")
    set_property(SOURCE mth_128defs_init.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_INIT)
    set_property(SOURCE mth_128defs_stats.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_STATS)

    set(MTH_INTRINSICS_SRCS
      mth_128defs.c
      mth_128defs_init.c
      mth_128defs_stats.c)
      libmath_add_object_library("${MTH_INTRINSICS_SRCS}" "${FLAGS}" "${DEFINITIONS}" "mth_intrinsics")

    set(SRCS
      mth_yintrinsics.c
      mth_zintrinsics.c
      kidnnt.c
      sincos.c
      sincosf.c)
  else()
    set_property(SOURCE dispatch.c APPEND_STRING PROPERTY COMPILE_FLAGS "-fno-builtin")
    set_property(SOURCE mth_xintrinsics.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mcpu=${LLVM_FLANG_CPU_TARGET} ")
    set_property(SOURCE mth_128defs.c mth_128defs_init.c mth_128defs_stats.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mcpu=${LLVM_FLANG_CPU_TARGET} ")
    set_property(SOURCE mth_128defs_init.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_INIT)
    set_property(SOURCE mth_128defs_stats.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_STATS)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z -Wl,combreloc")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z -Wl,stack-size=536870912")
    set(MTH_INTRINSICS_SRCS
      mth_128defs.c
      mth_128defs_init.c
      mth_128defs_stats.c)
    libmath_add_object_library("${MTH_INTRINSICS_SRCS}" "${FLAGS}" "${DEFINITIONS}" "mth_intrinsics")

    set(SRCS
      kidnnt.c
      sincos.c
      sincosf.c)
  endif()
else() # Generic Architecture
  set_property(SOURCE dispatch.c APPEND_STRING PROPERTY COMPILE_FLAGS "-fno-builtin")
  set_property(SOURCE mth_128defs_init.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_INIT)
  set_property(SOURCE mth_128defs_stats.c APPEND PROPERTY COMPILE_DEFINITIONS MTH_I_INTRIN_STATS)
  set(MTH_INTRINSICS_SRCS
    mth_128defs.c
    mth_128defs_init.c
    mth_128defs_stats.c)
  libmath_add_object_library("${MTH_INTRINSICS_SRCS}" "${FLAGS}" "${DEFINITIONS}" "mth_intrinsics")

  set(SRCS
    mth_yintrinsics.c
    mth_zintrinsics.c
    kidnnt.c
    sincos.c
    sincosf.c)
endif()

# Set source files
set(MTH_CMPLX_SRCS
  cabs.c
  cacos.c
  casin.c
  catan.c
  ccos.c
  ccosh.c
  cdabs.c
  cdacos.c
  cdasin.c
  cdatan.c
  cdcos.c
  cdcosh.c
  cddiv.c
  cddivd.c
  cdexp.c
  cdiv.c
  cdivr.c
  cdlog.c
  cdpowcd.c
  cdpowi.c
  cdpowk.c
  cdsin.c
  cdsinh.c
  cdsqrt.c
  cdtan.c
  cdtanh.c
  cexp.c
  clog.c
  cpowc.c
  cpowi.c
  cpowk.c
  csin.c
  csinh.c
  csqrt.c
  ctan.c
  ctanh.c)

set(MTH_CMPLX_SRCS_QUADFP
  cqabs.c
  cqdiv.c
  cqpowcq.c
  cqpowi.c
  cqpowk.c
  cqsin.c)

if(TARGET_SUPPORTS_QUADFP)
  LIST(APPEND MTH_CMPLX_SRCS ${MTH_CMPLX_SRCS_QUADFP})
endif()

set(SRCS
  ${SRCS}
  acosd.c
  amod.c
  asind.c
  atan2d.c
  atand.c
  bessel_j0.c
  bessel_j1.c
  bessel_jn.c
  bessel_y0.c
  bessel_y1.c
  bessel_yn.c
  ceil.c
  ${MTH_CMPLX_SRCS}
  cosd.c
  cvt.c
  dacosd.c
  dasind.c
  datan2d.c
  datand.c
  dbessel_j0.c
  dbessel_j1.c
  dbessel_jn.c
  dbessel_y0.c
  dbessel_y1.c
  dbessel_yn.c
  dceil.c
  dcosd.c
  dfloor.c
  dmod.c
  dpowi.c
  dpowk.c
  dsign.c
  dsind.c
  dtand.c
  erf.c
  erfc.c
  erfc_scaled.c
  erfc_scaledf.c
  erfcf.c
  erff.c
  floor.c
  fltfenv.c
  fltmanip.c
  fpcvt.c
  gamma.c
  gammaf.c
  hypot.c
  hypotf.c
  i2powi.c
  idnint.c
  ipowi.c
  knint.c
  kpowi.c
  kpowk.c
  log_gamma.c
  log_gammaf.c
  mod.c
  mth_vreturns.c
  mth_xintrinsics.c
  mthi64.c
  nint.c
  p_dpowi.c
  p_rpowi.c
  rpowi.c
  rpowk.c
  sign.c
  sind.c
  tand.c)

set(SRCS_QUADFP
  gammaq.c
  hypotq.c
  iqnint.c
  kqnint.c
  log_gammaq.c
  p_qpowi.c
  qacosd.c
  qasind.c
  qatan2d.c
  qatand.c
  qpowi.c
  qpowk.c
  qfloor.c
  qceil.c
  qmod.c
  qsign.c
  qsind.c
  qtand.c
  qcosd.c
  qerf.c
  qerfc.c
  qerfc_scaled.c)

if(TARGET_SUPPORTS_QUADFP)
  LIST(APPEND SRCS ${SRCS_QUADFP})
endif()

libmath_add_object_library("${SRCS}" "${FLAGS}" "${DEFINITIONS}" "")

set(DEFINITIONS_CMPLX ${DEFINITIONS} MTH_CMPLX_C99_ABI)
libmath_add_object_library("${MTH_CMPLX_SRCS}" "${FLAGS}" "${DEFINITIONS_CMPLX}" "common_mth_cmplx")

if(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "x86_64" AND NOT ${LIBPGMATH_WITH_GENERIC})
  # Generate mth_128mask.c
  set(TARGET_NAME "mth_128mask")
  add_custom_command(OUTPUT ${TARGET_NAME}.c DEPENDS ${LIBPGMATH_TOOLS_DIR}/mth_mask.awk PRE_BUILD
    COMMAND awk -v MAX_VREG_SIZE=128 -v TARGET=X8664 -f ${LIBPGMATH_TOOLS_DIR}/mth_mask.awk > ${TARGET_NAME}.c)
  add_custom_target(${TARGET_NAME} ALL DEPENDS "${TARGET_NAME}.c")
  if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Linux")
    set_property(SOURCE ${TARGET_NAME}.c APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core2")
  elseif(${LIBPGMATH_SYSTEM_NAME} MATCHES "Darwin|Windows")
    set_property(SOURCE ${TARGET_NAME}.c APPEND_STRING PROPERTY COMPILE_FLAGS "-msse2")
  endif()
  libmath_add_object_library("${TARGET_NAME}.c" "${FLAGS}" "${DEFINITIONS}" "${TARGET_NAME}_build")
  add_dependencies("${TARGET_NAME}_build" ${TARGET_NAME})

  # Generate mth_128generic.c
  set(TARGET_NAME "mth_128generic")
  add_custom_command(OUTPUT ${TARGET_NAME}.c DEPENDS ${LIBPGMATH_TOOLS_DIR}/mth_generic_frp.awk PRE_BUILD
    COMMAND awk -v MAX_VREG_SIZE=128 -v TARGET=X8664 -f ${LIBPGMATH_TOOLS_DIR}/mth_generic_frp.awk > ${TARGET_NAME}.c)
  add_custom_target(${TARGET_NAME} ALL DEPENDS "${TARGET_NAME}.c")
  if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Linux")
    set_property(SOURCE ${TARGET_NAME}.c APPEND_STRING PROPERTY COMPILE_FLAGS "-march=core2")
  elseif(${LIBPGMATH_SYSTEM_NAME} MATCHES "Darwin|Windows")
    set_property(SOURCE ${TARGET_NAME}.c APPEND_STRING PROPERTY COMPILE_FLAGS "-msse2")
  endif()
  libmath_add_object_library("${TARGET_NAME}.c" "${FLAGS}" "${DEFINITIONS}" "${TARGET_NAME}_build")
  add_dependencies("${TARGET_NAME}_build" ${TARGET_NAME})

  # Generate mth_256mask.c
  set(TARGET_NAME "mth_256mask")
  add_custom_command(OUTPUT ${TARGET_NAME}.c DEPENDS ${LIBPGMATH_TOOLS_DIR}/mth_mask.awk PRE_BUILD
    COMMAND awk -v MAX_VREG_SIZE=256 -v TARGET=X8664 -f ${LIBPGMATH_TOOLS_DIR}/mth_mask.awk > ${TARGET_NAME}.c)
  add_custom_target(${TARGET_NAME} ALL DEPENDS "${TARGET_NAME}.c")
  if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Linux")
    set_property(SOURCE ${TARGET_NAME}.c APPEND_STRING PROPERTY COMPILE_FLAGS "-march=sandybridge")
  elseif(${LIBPGMATH_SYSTEM_NAME} MATCHES "Darwin|Windows")
    set_property(SOURCE ${TARGET_NAME}.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mavx")
  endif()
  libmath_add_object_library("${TARGET_NAME}.c" "${FLAGS}" "${DEFINITIONS}" "${TARGET_NAME}_build")
  add_dependencies("${TARGET_NAME}_build" ${TARGET_NAME})

  # Generate mth_256generic.c
  set(TARGET_NAME "mth_256generic")
  add_custom_command(OUTPUT ${TARGET_NAME}.c DEPENDS ${LIBPGMATH_TOOLS_DIR}/mth_generic_frp.awk PRE_BUILD
    COMMAND awk -v MAX_VREG_SIZE=256 -v TARGET=X8664 -f ${LIBPGMATH_TOOLS_DIR}/mth_generic_frp.awk > ${TARGET_NAME}.c)
  add_custom_target(${TARGET_NAME} ALL DEPENDS "${TARGET_NAME}.c")
  if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Linux")
    set_property(SOURCE ${TARGET_NAME}.c APPEND_STRING PROPERTY COMPILE_FLAGS "-march=sandybridge")
  elseif(${LIBPGMATH_SYSTEM_NAME} MATCHES "Darwin|Windows")
    set_property(SOURCE ${TARGET_NAME}.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mavx")
  endif()
  libmath_add_object_library("${TARGET_NAME}.c" "${FLAGS}" "${DEFINITIONS}" "${TARGET_NAME}_build")
  add_dependencies("${TARGET_NAME}_build" ${TARGET_NAME})

  # Generate mth_512mask.c
  set(TARGET_NAME "mth_512mask")
  add_custom_command(OUTPUT ${TARGET_NAME}.c DEPENDS ${LIBPGMATH_TOOLS_DIR}/mth_mask.awk PRE_BUILD
    COMMAND awk -v MAX_VREG_SIZE=512 -v TARGET=X8664 -f ${LIBPGMATH_TOOLS_DIR}/mth_mask.awk > ${TARGET_NAME}.c)
  add_custom_target(${TARGET_NAME} ALL DEPENDS "${TARGET_NAME}.c")
  set_property(SOURCE ${TARGET_NAME}.c APPEND_STRING PROPERTY COMPILE_FLAGS "")
  if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Linux")
    set_property(SOURCE ${TARGET_NAME}.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mtune=knl -march=knl")
  elseif(${LIBPGMATH_SYSTEM_NAME} MATCHES "Darwin|Windows")
    set_property(SOURCE ${TARGET_NAME}.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mavx512f -mfma")
  endif()
  libmath_add_object_library("${TARGET_NAME}.c" "${FLAGS}" "${DEFINITIONS}" "${TARGET_NAME}_build")
  add_dependencies("${TARGET_NAME}_build" ${TARGET_NAME})

  # Generate mth_512generic.c
  set(TARGET_NAME "mth_512generic")
  add_custom_command(OUTPUT ${TARGET_NAME}.c DEPENDS ${LIBPGMATH_TOOLS_DIR}/mth_generic_frp.awk PRE_BUILD
    COMMAND awk -v MAX_VREG_SIZE=512 -v TARGET=X8664 -f ${LIBPGMATH_TOOLS_DIR}/mth_generic_frp.awk > ${TARGET_NAME}.c)
  add_custom_target(${TARGET_NAME} ALL DEPENDS "${TARGET_NAME}.c")
  if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Linux")
    set_property(SOURCE ${TARGET_NAME}.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mtune=knl -march=knl")
  elseif(${LIBPGMATH_SYSTEM_NAME} MATCHES "Darwin|Windows")
    set_property(SOURCE ${TARGET_NAME}.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mavx512f -mfma")
  endif()
  libmath_add_object_library("${TARGET_NAME}.c" "${FLAGS}" "${DEFINITIONS}" "${TARGET_NAME}_build")
  add_dependencies("${TARGET_NAME}_build" ${TARGET_NAME})

  # Generate mth_z2yy.c
  set(TARGET_NAME "mth_z2yy")
  add_custom_command(OUTPUT ${TARGET_NAME}.c DEPENDS ${LIBPGMATH_TOOLS_DIR}/mth_z2yy.awk PRE_BUILD
    COMMAND awk -v MAX_VREG_SIZE=512 -v TARGET=X8664 -f ${LIBPGMATH_TOOLS_DIR}/mth_z2yy.awk > ${TARGET_NAME}.c)
  add_custom_target(${TARGET_NAME} ALL DEPENDS "${TARGET_NAME}.c")
  if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Linux")
    set_property(SOURCE ${TARGET_NAME}.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mtune=knl -march=knl")
  elseif(${LIBPGMATH_SYSTEM_NAME} MATCHES "Darwin|Windows")
    set_property(SOURCE ${TARGET_NAME}.c APPEND_STRING PROPERTY COMPILE_FLAGS "-mavx512f -mfma")
  endif()
  libmath_add_object_library("${TARGET_NAME}.c" "${FLAGS}" "${DEFINITIONS}" "${TARGET_NAME}_build")
  add_dependencies("${TARGET_NAME}_build" ${TARGET_NAME})
elseif(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "ppc64le" AND NOT ${LIBPGMATH_WITH_GENERIC})
  # Generate mth_128mask.c
  set(TARGET_NAME "mth_128mask")
  add_custom_command(OUTPUT ${TARGET_NAME}.c DEPENDS ${LIBPGMATH_TOOLS_DIR}/mth_mask.awk PRE_BUILD
    COMMAND awk -v MAX_VREG_SIZE=128 -v TARGET=POWER -f ${LIBPGMATH_TOOLS_DIR}/mth_mask.awk > ${TARGET_NAME}.c)
  add_custom_target(${TARGET_NAME} ALL DEPENDS "${TARGET_NAME}.c")
  libmath_add_object_library("${TARGET_NAME}.c" "${FLAGS}" "${DEFINITIONS}" "${TARGET_NAME}_build")
  add_dependencies("${TARGET_NAME}_build" ${TARGET_NAME})

  # Generate mth_128generic.c
  set(TARGET_NAME "mth_128generic")
  add_custom_command(OUTPUT ${TARGET_NAME}.c DEPENDS ${LIBPGMATH_TOOLS_DIR}/mth_generic_frp.awk PRE_BUILD
    COMMAND awk -v MAX_VREG_SIZE=128 -v TARGET=POWER -f ${LIBPGMATH_TOOLS_DIR}/mth_generic_frp.awk > ${TARGET_NAME}.c)
  add_custom_target(${TARGET_NAME} ALL DEPENDS "${TARGET_NAME}.c")
  libmath_add_object_library("${TARGET_NAME}.c" "${FLAGS}" "${DEFINITIONS}" "${TARGET_NAME}_build")
  add_dependencies("${TARGET_NAME}_build" ${TARGET_NAME})
elseif(${LIBPGMATH_SYSTEM_PROCESSOR} MATCHES "aarch64" AND NOT ${LIBPGMATH_WITH_GENERIC})
  # Generate mth_128mask.cpp
  set(TARGET_NAME "mth_128mask")
  if(${LIBPGMATH_SYSTEM_NAME} MATCHES "Windows")
    add_custom_command(OUTPUT ${TARGET_NAME}.cpp DEPENDS ${LIBPGMATH_TOOLS_DIR}/mth_mask.awk PRE_BUILD
      COMMAND awk -v MAX_VREG_SIZE=128 -v TARGET=ARM64 -v TARGET_OS=WIN -f ${LIBPGMATH_TOOLS_DIR}/mth_mask.awk > ${TARGET_NAME}.cpp)
  else()
    add_custom_command(OUTPUT ${TARGET_NAME}.cpp DEPENDS ${LIBPGMATH_TOOLS_DIR}/mth_mask.awk PRE_BUILD
      COMMAND awk -v MAX_VREG_SIZE=128 -v TARGET=ARM64 -f ${LIBPGMATH_TOOLS_DIR}/mth_mask.awk > ${TARGET_NAME}.cpp)
  endif()
  add_custom_target(${TARGET_NAME} ALL DEPENDS "${TARGET_NAME}.cpp")
  libmath_add_object_library("${TARGET_NAME}.cpp" "${FLAGS}" "${DEFINITIONS}" "${TARGET_NAME}_build")
  add_dependencies("${TARGET_NAME}_build" ${TARGET_NAME})

  # Generate mth_128generic.c
  set(TARGET_NAME "mth_128generic")
  add_custom_command(OUTPUT ${TARGET_NAME}.c DEPENDS ${LIBPGMATH_TOOLS_DIR}/mth_generic_frp.awk PRE_BUILD
    COMMAND awk -v MAX_VREG_SIZE=128 -v TARGET=ARM64 -f ${LIBPGMATH_TOOLS_DIR}/mth_generic_frp.awk > ${TARGET_NAME}.c)
  add_custom_target(${TARGET_NAME} ALL DEPENDS "${TARGET_NAME}.c")
  libmath_add_object_library("${TARGET_NAME}.c" "${FLAGS}" "${DEFINITIONS}" "${TARGET_NAME}_build")
  add_dependencies("${TARGET_NAME}_build" ${TARGET_NAME})
else()
  # Generate mth_128mask.c
  set(TARGET_NAME "mth_128mask")
  add_custom_command(OUTPUT ${TARGET_NAME}.c DEPENDS ${LIBPGMATH_TOOLS_DIR}/mth_mask.awk PRE_BUILD
    COMMAND awk -v MAX_VREG_SIZE=128 -v TARGET=GENERIC -f ${LIBPGMATH_TOOLS_DIR}/mth_mask.awk > ${TARGET_NAME}.c)
  add_custom_target(${TARGET_NAME} ALL DEPENDS "${TARGET_NAME}.c")
  libmath_add_object_library("${TARGET_NAME}.c" "${FLAGS}" "${DEFINITIONS}" "${TARGET_NAME}_build")
  add_dependencies("${TARGET_NAME}_build" ${TARGET_NAME})

  # Generate mth_128generic.c
  set(TARGET_NAME "mth_128generic")
  add_custom_command(OUTPUT ${TARGET_NAME}.c DEPENDS ${LIBPGMATH_TOOLS_DIR}/mth_generic_frp.awk PRE_BUILD
    COMMAND awk -v MAX_VREG_SIZE=128 -v TARGET=GENERIC -f ${LIBPGMATH_TOOLS_DIR}/mth_generic_frp.awk > ${TARGET_NAME}.c)
  add_custom_target(${TARGET_NAME} ALL DEPENDS "${TARGET_NAME}.c")
  libmath_add_object_library("${TARGET_NAME}.c" "${FLAGS}" "${DEFINITIONS}" "${TARGET_NAME}_build")
  add_dependencies("${TARGET_NAME}_build" ${TARGET_NAME})
endif()

# Generate tmp-mth_statsdefs.h
set(TARGET_NAME "tmp-mth_statsdefs")
add_custom_command(OUTPUT ${TARGET_NAME}.h PRE_BUILD
  COMMAND ${SH_PROGRAM} "${LIBPGMATH_TOOLS_DIR}/${TARGET_NAME}_${PROCESSOR}.sh" ${CMAKE_CURRENT_SOURCE_DIR} ${TARGET_NAME}.h ${TARGET_SUPPORTS_QUADFP})

add_custom_target(${TARGET_NAME} ALL DEPENDS "${TARGET_NAME}.h")
libmath_add_object_library("dispatch.c" "${FLAGS}" "${DEFINITIONS}" "dispatch")
add_dependencies(dispatch ${TARGET_NAME} tmp-mth_alldefs)
add_dependencies(dispatch ${TARGET_NAME} tmp-mth_statsdefs)
target_include_directories(dispatch
  BEFORE
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../${PROCESSOR}/math_tables
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/../generic/math_tables)
