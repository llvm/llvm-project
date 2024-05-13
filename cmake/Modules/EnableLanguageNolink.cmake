macro(llvm_enable_language_nolink)
  # Set CMAKE_TRY_COMPILE_TARGET_TYPE to STATIC_LIBRARY to disable linking
  # in the compiler sanity checks. When bootstrapping the toolchain,
  # the toolchain itself is still incomplete and sanity checks that include
  # linking may fail.
  set(__SAVED_TRY_COMPILE_TARGET_TYPE ${CMAKE_TRY_COMPILE_TARGET_TYPE})
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  enable_language(${ARGV})
  set(CMAKE_TRY_COMPILE_TARGET_TYPE ${__SAVED_TRY_COMPILE_TARGET_TYPE})
  unset(__SAVED_TRY_COMPILE_TARGET_TYPE)
endmacro()
