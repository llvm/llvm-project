// RUN: env MACOSX_DEPLOYMENT_TARGET=10.1 \
// RUN:   %clang -target i386-apple-darwin9 -DTEST0 -E %s
#ifdef TEST0
#if __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ != 1010
#error Invalid version
#endif
#endif

// RUN: env IPHONEOS_DEPLOYMENT_TARGET=2.0 \
// RUN:   %clang -target i386-apple-darwin9 -DTEST1 -E %s
#ifdef TEST1
#if __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__ != 20000
#error Invalid version
#endif
#endif

// RUN: env IPHONEOS_DEPLOYMENT_TARGET=2.3.1 \
// RUN:   %clang -target i386-apple-darwin9 -DTEST2 -E %s
#ifdef TEST2
#if __ENVIRONMENT_IPHONE_OS_VERSION_MIN_REQUIRED__ != 20301
#error Invalid version
#endif
#endif

// RUN: env MACOSX_DEPLOYMENT_TARGET=10.4.10 \
// RUN:   %clang -target i386-apple-darwin9 -DTEST3 -E %s
#ifdef TEST3
#if __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ != 1049
#error Invalid version
#endif
#endif
