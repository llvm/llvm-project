// RUN: %clang_cc1 -E -x c %s > /dev/tty
// The original bug requires UNIX line endings to trigger.
// The original bug triggers only when outputting directly to console.
// REQUIRES: console

// Make sure clang does not crash during preprocessing

#define M0 extern int x;
#define M2  M0  M0  M0  M0
#define M4  M2  M2  M2  M2
#define M6  M4  M4  M4  M4
#define M8  M6  M6  M6  M6
#define M10 M8  M8  M8  M8
#define M12 M10 M10 M10 M10
#define M14 M12 M12 M12 M12

M14
