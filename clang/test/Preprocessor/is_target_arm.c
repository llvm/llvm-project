// RUN: %clang_cc1 -fsyntax-only -triple thumbv7--windows-msvc19.11.0 -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple armv7--windows-msvc19.11.0 -DARM -verify %s
// expected-no-diagnostics

// ARM does match arm and thumb.
#if !__is_target_arch(arm)
  #error "mismatching arch"
#endif

#if __is_target_arch(armeb) || __is_target_arch(armebv7) || __is_target_arch(thumbeb) || __is_target_arch(thumbebv7)
  #error "mismatching arch"
#endif

// ARMV7 does match armv7 and thumbv7.
#if !__is_target_arch(armv7)
  #error "mismatching arch"
#endif

// ARMV6 does not match armv7 or thumbv7.
#if __is_target_arch(armv6)
  #error "mismatching arch"
#endif

#if __is_target_arch(arm64)
  #error "mismatching arch"
#endif

#ifndef ARM

// Allow checking for precise arch + subarch.
#if !__is_target_arch(thumbv7)
  #error "mismatching arch"
#endif

// But also allow checking for the arch without subarch.
#if !__is_target_arch(thumb)
  #error "mismatching arch"
#endif

// Same arch with a different subarch doesn't match.
#if __is_target_arch(thumbv6)
  #error "mismatching arch"
#endif

#else

#if __is_target_arch(thumbv7) || __is_target_arch(thumb)
  #error "mismatching arch"
#endif

#endif
