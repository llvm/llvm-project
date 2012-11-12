// RUN: %clang -fsyntax-only -fmacro-backtrace-limit=0 %s 2>&1 | FileCheck %s

#define FOO 1+"hi" 
#define BAR FOO
#define BAZ BAR
#define QUZ BAZ
#define TAZ QUZ
#define ABA TAZ
#define BAB ABA
#define ZAZ BAB
#define WAZ ZAZ
#define DROOL WAZ
#define FOOL DROOL

FOOL

// CHECK: :15:1: error: expected identifier or '('
// CHECK: FOOL
// CHECK: ^
// CHECK: :13:14: note: expanded from macro 'FOOL'
// CHECK: #define FOOL DROOL
// CHECK:              ^
// CHECK: :12:15: note: expanded from macro 'DROOL'
// CHECK: #define DROOL WAZ
// CHECK:               ^
// CHECK: :11:13: note: expanded from macro 'WAZ'
// CHECK: #define WAZ ZAZ
// CHECK:             ^
// CHECK: :10:13: note: expanded from macro 'ZAZ'
// CHECK: #define ZAZ BAB
// CHECK:             ^
// CHECK: :9:13: note: expanded from macro 'BAB'
// CHECK: #define BAB ABA
// CHECK:             ^
// CHECK: :8:13: note: expanded from macro 'ABA'
// CHECK: #define ABA TAZ
// CHECK:             ^
// CHECK: :7:13: note: expanded from macro 'TAZ'
// CHECK: #define TAZ QUZ
// CHECK:             ^
// CHECK: :6:13: note: expanded from macro 'QUZ'
// CHECK: #define QUZ BAZ
// CHECK:             ^
// CHECK: :5:13: note: expanded from macro 'BAZ'
// CHECK: #define BAZ BAR
// CHECK:             ^
// CHECK: :4:13: note: expanded from macro 'BAR'
// CHECK: #define BAR FOO
// CHECK:             ^
// CHECK: :3:13: note: expanded from macro 'FOO'
// CHECK: #define FOO 1+"hi" 
// CHECK:             ^
// CHECK: 1 error generated.
