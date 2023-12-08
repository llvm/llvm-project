// RUN: %clang_cc1 -std=gnu++17 -verify %s

// expected-no-diagnostics

typedef const struct __CFString * CFStringRef;

extern "C" {
  typedef const struct __attribute__((objc_bridge(NSString))) __CFString * CFStringRef;
  typedef struct __attribute__((objc_bridge_mutable(NSMutableString))) __CFString * CFMutableStringRef;
}

@interface NSString @end
@interface NSMutableString : NSString @end

void CFStringGetLength(CFStringRef theString);

int main() {
  CFStringGetLength((__bridge CFStringRef)(NSString *)0);
}
