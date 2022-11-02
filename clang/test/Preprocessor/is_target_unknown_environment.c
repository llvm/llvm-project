// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-macos12 -verify %s

// expected-no-diagnostics

#if !__is_target_environment(unknown)
#error "mismatching environment"
#endif

#if __is_target_environment(simulator) || __is_target_environment(SIMULATOR)
#error "mismatching environment"
#endif

#if __is_target_environment(invalidEnv)
#error "invalid environment must not be matched"
#endif
