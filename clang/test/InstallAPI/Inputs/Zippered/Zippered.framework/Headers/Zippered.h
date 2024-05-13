#if !__is_target_environment(macabi)
typedef int MyType;
#else
typedef float MyType;
#endif

extern MyType invalidAPI();

#define OS_AVAILABLE(_target, _availability)                                   \
  __attribute__((availability(_target, _availability)))
extern int macOSAPI() OS_AVAILABLE(macos, introduced=10.14) OS_AVAILABLE(ios, unavailable);
extern int iOSAPI() OS_AVAILABLE(ios, introduced=12.0) OS_AVAILABLE(macos, unavailable);
extern int commonAPI() OS_AVAILABLE(macos, introduced=10.14) OS_AVAILABLE(ios, introduced=12.0);

extern int obsoletedMacOSAPI() OS_AVAILABLE(macos, obsoleted=10.14) OS_AVAILABLE(ios, unavailable);

#if !__is_target_environment(macabi)
extern int macOSAPI2() OS_AVAILABLE(macos, introduced = 10.14)
    OS_AVAILABLE(ios, unavailable);
#endif
