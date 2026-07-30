// RUN: %clang_cc1 %s
#ifndef __has_feature
#error Should have __has_feature
#endif

#if !__has_feature(objc_nonfragile_abi)
#error Non-fragile ABI used for compilation but feature macro not set.
#endif

#if !__has_feature(objc_weak_class)
#error objc_weak_class should be enabled with nonfragile abi
#endif
