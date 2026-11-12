@class NSError;

#pragma clang assume_nonnull begin

#ifdef USE_MUTABLE
typedef struct __attribute__((objc_bridge_mutable(NSError))) __CFError * CFErrorRef;
#else
typedef struct __attribute__((objc_bridge(NSError))) __CFError * CFErrorRef;
#endif

void func1(CFErrorRef *error);

#pragma clang assume_nonnull end
