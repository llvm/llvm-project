// RUN: %clang_cc1 -triple armv6-apple-ios3.0.0 -dM -E -o %t %s
// RUN: grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t | grep '30000' | count 1
// RUN: not grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t
// RUN: %clang_cc1 -triple armv6-apple-ios2.0.0 -dM -E -o %t %s
// RUN: grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t | grep '20000' | count 1
// RUN: not grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t
// RUN: %clang_cc1 -triple armv6-apple-ios2.2.0 -dM -E -o %t %s
// RUN: grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t | grep '20200' | count 1
// RUN: not grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t
// RUN: %clang_cc1 -triple armv6-apple-ios2.3.1 -dM -E -o %t %s
// RUN: grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t | grep '20301' | count 1
// RUN: not grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t
// RUN: %clang_cc1 -triple armv7-apple-ios10.1.2 -dM -E -o %t %s
// RUN: grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t | grep '100102' | count 1
// RUN: %clang_cc1 -triple i386-apple-macosx10.4.0 -dM -E -o %t %s
// RUN: grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t | grep '1040' | count 1
// RUN: not grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t
// RUN: not grep '__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__' %t
// RUN: not grep '__ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__' %t
// RUN: %clang_cc1 -triple i386-apple-macosx10.4.10 -dM -E -o %t %s
// RUN: grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t | grep '1049' | count 1
// RUN: not grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t
// RUN: %clang_cc1 -triple i386-apple-macosx10.5.0 -dM -E -o %t %s
// RUN: grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t | grep '1050' | count 1
// RUN: not grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t
// RUN: %clang_cc1 -triple i386-apple-macosx10.6.0 -dM -E -o %t %s
// RUN: grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t | grep '1060' | count 1
// RUN: not grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t
// RUN: %clang_cc1 -triple x86_64-apple-macosx10.10.0 -dM -E -o %t %s
// RUN: grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t | grep '101000' | count 1
// RUN: not grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t

// RUN: %clang_cc1 -triple arm64-apple-tvos8.3 -dM -E -o %t %s
// RUN: grep '__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__' %t | grep '80300' | count 1
// RUN: not grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t
// RUN: not grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t
// RUN: %clang_cc1 -triple arm64-apple-tvos10.2.3 -dM -E -o %t %s
// RUN: grep '__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__' %t | grep '100203' | count 1

// RUN: %clang_cc1 -triple x86_64-apple-tvos8.3 -dM -E -o %t %s
// RUN: grep '__ENVIRONMENT_TV_OS_VERSION_MIN_REQUIRED__' %t | grep '80300' | count 1
// RUN: not grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t
// RUN: not grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t

// RUN: %clang_cc1 -triple armv7k-apple-watchos2.1 -dM -E -o %t %s
// RUN: grep '__ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__' %t | grep '20100' | count 1
// RUN: not grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t
// RUN: not grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t

// RUN: %clang_cc1 -triple i386-apple-watchos2.1 -dM -E -o %t %s
// RUN: grep '__ENVIRONMENT_WATCH_OS_VERSION_MIN_REQUIRED__' %t | grep '20100' | count 1
// RUN: not grep '__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__' %t
// RUN: not grep '__ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__' %t
