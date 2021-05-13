# ------------------------------------------------------------------------------
# Architecture definitions
# ------------------------------------------------------------------------------

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^mips")
  set(LIBC_TARGET_ARCHITECTURE_IS_MIPS TRUE)
  set(LIBC_TARGET_ARCHITECTURE "mips")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
  set(LIBC_TARGET_ARCHITECTURE_IS_ARM TRUE)
  set(LIBC_TARGET_ARCHITECTURE "arm")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64")
  set(LIBC_TARGET_ARCHITECTURE_IS_AARCH64 TRUE)
  set(LIBC_TARGET_ARCHITECTURE "aarch64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
  set(LIBC_TARGET_ARCHITECTURE_IS_X86 TRUE)
  set(LIBC_TARGET_ARCHITECTURE "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc|ppc)")
  set(LIBC_TARGET_ARCHITECTURE_IS_POWER TRUE)
  set(LIBC_TARGET_ARCHITECTURE "power")
else()
  message(FATAL_ERROR "Unsupported processor ${CMAKE_SYSTEM_PROCESSOR}")
endif()
