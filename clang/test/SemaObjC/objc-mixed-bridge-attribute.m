// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics
// rdar://17238954

typedef const struct __attribute__((objc_bridge(NSAttributedString))) __CFAttributedString *CFAttributedStringRef;

typedef struct __attribute__((objc_bridge_mutable(NSMutableAttributedString))) __CFAttributedString *CFMutableAttributedStringRef;

@interface NSAttributedString
@end

@interface NSMutableAttributedString
@end

struct __CFAttributedString {
};

void Test1(CFAttributedStringRef attrStr, CFMutableAttributedStringRef mutable_attrStr)
{
  id x = (NSAttributedString *) attrStr;
  id x1 =(NSAttributedString *) mutable_attrStr;
  id x2 = (NSMutableAttributedString *) attrStr;
  id x3 = (NSMutableAttributedString *) mutable_attrStr;
}

void Test2(NSAttributedString *ns_attrStr, NSMutableAttributedString *ns_mutable_attr_Str) {
  CFAttributedStringRef cfsr = (CFAttributedStringRef) ns_attrStr;
  CFMutableAttributedStringRef cfsr1 = (CFMutableAttributedStringRef) ns_attrStr;
  CFAttributedStringRef cfsr2 = (CFAttributedStringRef) ns_mutable_attr_Str;
  CFMutableAttributedStringRef cfsr3 = (CFMutableAttributedStringRef) ns_mutable_attr_Str;
}

// Tests with no definition declaration for struct __NDCFAttributedString.
typedef const struct __attribute__((objc_bridge(NSAttributedString))) __NDCFAttributedString *NDCFAttributedStringRef;

typedef struct __attribute__((objc_bridge_mutable(NSMutableAttributedString))) __NDCFAttributedString *NDCFMutableAttributedStringRef;

void Test3(NDCFAttributedStringRef attrStr, NDCFMutableAttributedStringRef mutable_attrStr)
{
  id x = (NSAttributedString *) attrStr;
  id x1 =(NSAttributedString *) mutable_attrStr;
  id x2 = (NSMutableAttributedString *) attrStr;
  id x3 = (NSMutableAttributedString *) mutable_attrStr;
}

void Test4(NSAttributedString *ns_attrStr, NSMutableAttributedString *ns_mutable_attr_Str) {
  NDCFAttributedStringRef cfsr = (NDCFAttributedStringRef) ns_attrStr;
  NDCFMutableAttributedStringRef cfsr1 = (NDCFMutableAttributedStringRef) ns_attrStr;
  NDCFAttributedStringRef cfsr2 = (NDCFAttributedStringRef) ns_mutable_attr_Str;
  NDCFMutableAttributedStringRef cfsr3 = (NDCFMutableAttributedStringRef) ns_mutable_attr_Str;
}
