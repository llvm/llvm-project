// RUN: %clang_cc1 -fsyntax-only -verify < %s
// expected-no-diagnostics
typedef float CGFloat;
typedef struct _NSPoint { CGFloat x; CGFloat y; } NSPoint;
typedef struct _NSSize { CGFloat width; CGFloat height; } NSSize;
typedef struct _NSRect { NSPoint origin; NSSize size; } NSRect;

extern const NSPoint NSZeroPoint;

extern NSSize canvasSize(void);
void func(void) {
   const NSRect canvasRect = { NSZeroPoint, canvasSize() };
}
