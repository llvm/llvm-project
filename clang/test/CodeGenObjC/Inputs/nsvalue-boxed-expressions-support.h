#ifndef NSVALUE_BOXED_EXPRESSIONS_SUPPORT_H
#define NSVALUE_BOXED_EXPRESSIONS_SUPPORT_H

#define BOXABLE __attribute__((objc_boxable))

typedef unsigned long NSUInteger;
typedef double CGFloat;

typedef struct BOXABLE _NSRange {
    NSUInteger location;
    NSUInteger length;
} NSRange;

typedef struct BOXABLE _NSPoint {
    CGFloat x;
    CGFloat y;
} NSPoint;

typedef struct BOXABLE _NSSize {
    CGFloat width;
    CGFloat height;
} NSSize;

typedef struct BOXABLE _NSRect {
    NSPoint origin;
    NSSize size;
} NSRect;

struct CGPoint {
  CGFloat x;
  CGFloat y;
};
typedef struct BOXABLE CGPoint CGPoint;

struct CGSize {
  CGFloat width;
  CGFloat height;
};
typedef struct BOXABLE CGSize CGSize;

struct CGRect {
  CGPoint origin;
  CGSize size;
};
typedef struct BOXABLE CGRect CGRect;

struct NSEdgeInsets {
  CGFloat top;
  CGFloat left;
  CGFloat bottom;
  CGFloat right;
};
typedef struct BOXABLE NSEdgeInsets NSEdgeInsets;

@interface NSValue

+ (NSValue *)valueWithBytes:(const void *)value objCType:(const char *)type;

@end

NSRange getRange(void);

#endif // NSVALUE_BOXED_EXPRESSIONS_SUPPORT_H
