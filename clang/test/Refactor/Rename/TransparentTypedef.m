#define NS_ENUM(_name, _type) enum _name:_type _name; enum _name : _type
// CHECK1: Renaming 1 symbols
// CHECK1-NEXT: 'c:@E@AnotherEnum'
typedef NS_ENUM(AnotherEnum, int) { // CHECK1: rename [[@LINE]]:17 -> [[@LINE]]:28
  AnotherEnumFirst = 0,
};
AnotherEnum anotherT; // CHECK1: rename [[@LINE]]:1 -> [[@LINE]]:12
enum AnotherEnum anotherE; // CHECK1: rename [[@LINE]]:6 -> [[@LINE]]:17

// RUN: clang-refactor-test rename-initiate -at=%s:4:17 -at=%s:7:1 -at=%s:8:6 -new-name=foo -dump-symbols %s | FileCheck --check-prefix=CHECK1 %s

#define TRANSPARENT(_name) struct _name _name; struct _name
#define OPAQUE(_name) struct _name *_name; struct _name

// CHECK2: Renaming 1 symbols
// CHECK2-NEXT: 'c:@S@AStruct'
typedef TRANSPARENT(AStruct) { // CHECK2: rename [[@LINE]]:21 -> [[@LINE]]:28
  int x;
};

AStruct aStructT; // CHECK2: rename [[@LINE]]:1 -> [[@LINE]]:8
struct AStruct aStructS; // CHECK2: rename [[@LINE]]:8 -> [[@LINE]]:15

// RUN: clang-refactor-test rename-initiate -at=%s:17:21 -at=%s:21:1 -at=%s:22:8 -new-name=foo -dump-symbols %s | FileCheck --check-prefix=CHECK2 %s

// CHECK3: Renaming 1 symbols
// CHECK3-NEXT: 'c:{{.*}}TransparentTypedef.m@T@Separate'
// CHECK4: Renaming 1 symbols
// CHECK4-NEXT: 'c:@S@Separate'

typedef OPAQUE(Separate) { // CHECK3: rename [[@LINE]]:16 -> [[@LINE]]:24
  int x; // CHECK4: rename [[@LINE-1]]:16 -> [[@LINE-1]]:24
};


Separate separateT; // CHECK3: rename [[@LINE]]:1 -> [[@LINE]]:9
struct Separate separateE;  // CHECK4: rename [[@LINE]]:8 -> [[@LINE]]:16

// RUN: clang-refactor-test rename-initiate -at=%s:36:1 -new-name=foo -dump-symbols %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-initiate -at=%s:31:16 -at=%s:37:8 -new-name=foo -dump-symbols %s | FileCheck --check-prefix=CHECK4 %s

#include "Inputs/TransparentEnum.h"

// CHECK5: 'c:@E@AnotherEnum2'
typedef TRANSPARENT_ENUM(AnotherEnum2, int) { // CHECK5: rename [[@LINE]]:26 -> [[@LINE]]:38
  EnumThing = 0, // CHECK6: [[@LINE]]:3 -> [[@LINE]]:12
};
// RUN: clang-refactor-test rename-initiate -at=%s:45:26 -new-name=foo -dump-symbols %s | FileCheck --check-prefix=CHECK5 %s
// RUN: clang-refactor-test rename-initiate -at=%s:46:3 -new-name=foo %s | FileCheck --check-prefix=CHECK6 %s
