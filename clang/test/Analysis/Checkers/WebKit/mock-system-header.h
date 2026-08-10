#pragma clang system_header

template <typename T, typename CreateFunction>
void callMethod(CreateFunction createFunction) {
  createFunction()->method();
}

template <typename T, typename CreateFunction>
inline void localVar(CreateFunction createFunction) {
  T* obj = createFunction();
  obj->method();
}

template <typename T>
struct MemberVariable {
    T* obj { nullptr };
};

typedef struct sqlite3 sqlite3;

typedef unsigned char uint8_t;

enum os_log_type_t : uint8_t {
    OS_LOG_TYPE_DEFAULT = 0x00,
    OS_LOG_TYPE_INFO = 0x01,
    OS_LOG_TYPE_DEBUG = 0x02,
    OS_LOG_TYPE_ERROR = 0x10,
    OS_LOG_TYPE_FAULT = 0x11,
};

typedef struct os_log_s *os_log_t;
os_log_t os_log_create(const char *subsystem, const char *category);
void os_log_msg(os_log_t oslog, os_log_type_t type, const char *msg, ...);

typedef const struct __attribute__((objc_bridge(NSString))) __CFString * CFStringRef;

extern CFStringRef const kCFURLTagNamesKey;

#ifdef __OBJC__
@class NSString;
@interface SystemObject {
  NSString *ns_string;
  CFStringRef cf_string;
}
@end

typedef NSString *NSNotificationName;
extern "C" NSNotificationName NSApplicationDidBecomeActiveNotification;

#endif
