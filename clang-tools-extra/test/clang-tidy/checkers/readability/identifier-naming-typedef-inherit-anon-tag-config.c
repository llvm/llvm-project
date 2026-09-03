// RUN: %check_clang_tidy -std=c17 %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.TypedefInheritAnonTagConfig: true, \
// RUN:     readability-identifier-naming.EnumCase: CamelCase, \
// RUN:     readability-identifier-naming.StructCase: lower_case, \
// RUN:     readability-identifier-naming.TypedefCase: camelBack, \
// RUN:   }}'

typedef enum { EV_ANON } my_enum;
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: invalid case style for enum 'my_enum' [readability-identifier-naming]
// CHECK-FIXES: typedef enum { EV_ANON } MyEnum;

typedef struct { int Field; } My_Struct;
// CHECK-MESSAGES: :[[@LINE-1]]:31: warning: invalid case style for struct 'My_Struct' [readability-identifier-naming]
// CHECK-FIXES: typedef struct { int Field; } my_struct;

// The tag has a name of its own, so the typedef style still applies.
typedef struct data { int Field; } my_data;
// CHECK-MESSAGES: :[[@LINE-1]]:36: warning: invalid case style for typedef 'my_data' [readability-identifier-naming]
// CHECK-FIXES: typedef struct data { int Field; } myData;
