#if __has_feature(address_sanitizer)
#define HAS_ASAN 1
#else
#define HAS_ASAN 0
#endif
