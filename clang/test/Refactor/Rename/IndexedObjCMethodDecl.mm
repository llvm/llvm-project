@implementation foo
+(some_type_t)a_200:(void)a_200 name:(int[1 + 2 - 3])z_Z_42 usingThing:(some_type_t)world method:(void)test
class:(int[1 + 2 - 3])method __attribute__((eval { int x = 0 + 1; })) method:(({}))method {
  const Object & piece = 12;
}
// CHECK1: [[@LINE-4]]:15 -> [[@LINE-4]]:20, [[@LINE-4]]:33 -> [[@LINE-4]]:37, [[@LINE-4]]:61 -> [[@LINE-4]]:71, [[@LINE-4]]:91 -> [[@LINE-4]]:97, [[@LINE-3]]:1 -> [[@LINE-3]]:6, [[@LINE-3]]:71 -> [[@LINE-3]]:77
+(some_type_t)world:(BOOL)withSomething class:(const Object &)name struct:(some_type_t)method __attribute__((test()))a_200:(int)name
a_200:(({}))perform withSomething:(Object * (^)(BOOL, Object *))onEntity
{
  const Object & class = globalArray[i];
}
// CHECK2: [[@LINE-5]]:15 -> [[@LINE-5]]:20, [[@LINE-5]]:41 -> [[@LINE-5]]:46, [[@LINE-5]]:68 -> [[@LINE-5]]:74, [[@LINE-5]]:118 -> [[@LINE-5]]:123, [[@LINE-4]]:1 -> [[@LINE-4]]:6, [[@LINE-4]]:21 -> [[@LINE-4]]:34
-(some_type_t)struct:(int[1 + 2 - 3])bar
class:(int[1 + 2 - 3])foo
part:(void)piece
{
  int onEntity = "]";
}
// CHECK3: [[@LINE-6]]:15 -> [[@LINE-6]]:21, [[@LINE-5]]:1 -> [[@LINE-5]]:6, [[@LINE-4]]:1 -> [[@LINE-4]]:5
-(some_type_t)test:(^ { })a_200 world:(BOOL)onEntity a_200:(BOOL)perform
withSomething:(({}))method
{
  [self part: [self z_Z_42: "]" world: ("]")] world: globalArray[i] class: "string" test: globalArray[i]];
}
// CHECK4: [[@LINE-5]]:15 -> [[@LINE-5]]:19, [[@LINE-5]]:33 -> [[@LINE-5]]:38, [[@LINE-5]]:54 -> [[@LINE-5]]:59, [[@LINE-4]]:1 -> [[@LINE-4]]:14
-(int)a_200:(BOOL)bar piece:(Object *)perform class:(^ { })onEntity {
  [globalObject send: [self perform: ']' object: [] () {    ([self name: ']' a_200: ']']) other: 42];
  } foo: ']'];
}
// CHECK5: [[@LINE-4]]:7 -> [[@LINE-4]]:12, [[@LINE-4]]:23 -> [[@LINE-4]]:28, [[@LINE-4]]:47 -> [[@LINE-4]]:52
@end
// RUN: clang-refactor-test rename-indexed-file -name=a_200:name:usingThing:method:class:method -new-name=withSomething:struct:class:onEntity:withSomething:test -indexed-file=%s -indexed-at=2:15 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK1 %s
// RUN: clang-refactor-test rename-indexed-file -name=world:class:struct:a_200:a_200:withSomething -new-name=onEntity:a_200:perform:onEntity:usingThing:onEntity -indexed-file=%s -indexed-at=7:15 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-indexed-file -name=struct:class:part -new-name=z_Z_42:name:usingThing -indexed-file=%s -indexed-at=13:15 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK3 %s
// RUN: clang-refactor-test rename-indexed-file -name=test:world:a_200:withSomething -new-name=part:withSomething:test:struct -indexed-file=%s -indexed-at=20:15 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test rename-indexed-file -name=a_200:piece:class -new-name=a_200:a_200:piece -indexed-file=%s -indexed-at=26:7 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK5 %s

@interface method
+(some_type_t)bar:(BOOL)bar bar:(Object *)usingThing struct:(int[1 + 2 - 3])usingThing
perform:(BOOL)onEntity ;
// CHECK6: [[@LINE-2]]:15 -> [[@LINE-2]]:18, [[@LINE-2]]:29 -> [[@LINE-2]]:32, [[@LINE-2]]:54 -> [[@LINE-2]]:60, [[@LINE-1]]:1 -> [[@LINE-1]]:8
+(void)struct:(^ { })part __attribute__((test()))foo:(^ { })part part:(void)class __attribute__((eval { int x = 0 + 1; })) onEntity:(Object *)method foo:(const Object &)class /*comment*/ ;
// CHECK7: [[@LINE-1]]:8 -> [[@LINE-1]]:14, [[@LINE-1]]:50 -> [[@LINE-1]]:53, [[@LINE-1]]:66 -> [[@LINE-1]]:70, [[@LINE-1]]:124 -> [[@LINE-1]]:132, [[@LINE-1]]:150 -> [[@LINE-1]]:153
@end
// RUN: clang-refactor-test rename-indexed-file -name=bar:bar:struct:perform -new-name=bar:withSomething:test:foo -indexed-file=%s -indexed-at=39:15 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK6 %s
// RUN: clang-refactor-test rename-indexed-file -name=struct:foo:part:onEntity:foo -new-name=z_Z_42:part:world:piece:perform -indexed-file=%s -indexed-at=42:8 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK7 %s

@interface foo
+(BOOL)class:(Object *)method struct:(void)method
perform:(Object *)class
;
// CHECK8: [[@LINE-3]]:8 -> [[@LINE-3]]:13, [[@LINE-3]]:31 -> [[@LINE-3]]:37, [[@LINE-2]]:1 -> [[@LINE-2]]:8
+(some_type_t)usingThing:(({}))struct withSomething:(const Object &)z_Z_42 method:(some_type_t)class
a_200:(int)a_200 ;
// CHECK9: [[@LINE-2]]:15 -> [[@LINE-2]]:25, [[@LINE-2]]:39 -> [[@LINE-2]]:52, [[@LINE-2]]:76 -> [[@LINE-2]]:82, [[@LINE-1]]:1 -> [[@LINE-1]]:6
+(Object *)a_200:(const Object &)class //comment
class:(BOOL)bar usingThing:(void (*)(some_type_t, some_type_t))name
bar:(BOOL)onEntity method:(void)method
part:(BOOL)usingThing /*comment*/ ;
// CHECK10: [[@LINE-4]]:12 -> [[@LINE-4]]:17, [[@LINE-3]]:1 -> [[@LINE-3]]:6, [[@LINE-3]]:17 -> [[@LINE-3]]:27, [[@LINE-2]]:1 -> [[@LINE-2]]:4, [[@LINE-2]]:20 -> [[@LINE-2]]:26, [[@LINE-1]]:1 -> [[@LINE-1]]:5
@end
// RUN: clang-refactor-test rename-indexed-file -name=class:struct:perform -new-name=withSomething:foo:a_200 -indexed-file=%s -indexed-at=49:8 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK8 %s
// RUN: clang-refactor-test rename-indexed-file -name=usingThing:withSomething:method:a_200 -new-name=piece:bar:test:perform -indexed-file=%s -indexed-at=53:15 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK9 %s
// RUN: clang-refactor-test rename-indexed-file -name=a_200:class:usingThing:bar:method:part -new-name=a_200:piece:class:z_Z_42:piece:world -indexed-file=%s -indexed-at=56:12 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK10 %s

@implementation onEntity <object, onEntity, world>
-(const Object &)test:(some_type_t (*)(some_type_t))z_Z_42 //comment
perform:(Object *)name /*comment*/ method:(int)bar __attribute__((eval { int x = 0 + 1; })) part:(Object *)name z_Z_42:(BOOL)a_200 test:(^ { })usingThing __attribute__((test())){
  call() = [self test: "]" z_Z_42: ^  {    [_undef_ivar foo: "string" piece: @{ @1, @3 } perform: @"string literal" world: ']'];
  } == "]" a_200: 12/*comment*/ object: 12 test: 12];
}
// CHECK11: [[@LINE-5]]:18 -> [[@LINE-5]]:22, [[@LINE-4]]:1 -> [[@LINE-4]]:8, [[@LINE-4]]:36 -> [[@LINE-4]]:42, [[@LINE-4]]:93 -> [[@LINE-4]]:97, [[@LINE-4]]:113 -> [[@LINE-4]]:119, [[@LINE-4]]:132 -> [[@LINE-4]]:136
-(const Object &)z_Z_42:(({}))usingThing onEntity:(^ { })world
a_200:(const Object &)withSomething __attribute__((test()))onEntity:(({}))onEntity {
  [_undef_ivar object: globalArray[i] perform: ^  {    globalArray[12] = [_undef_ivar class: "]" onEntity: []  {
  } bar: "string" < ^  {
  } foo: "string" piece: @{ @1, @3 }];
  } test: /*]*/  method: globalArray[i]
];
}
// CHECK12: [[@LINE-8]]:18 -> [[@LINE-8]]:24, [[@LINE-8]]:42 -> [[@LINE-8]]:50, [[@LINE-7]]:1 -> [[@LINE-7]]:6, [[@LINE-7]]:60 -> [[@LINE-7]]:68
@end
// RUN: clang-refactor-test rename-indexed-file -name=test:perform:method:part:z_Z_42:test -new-name=a_200:piece:onEntity:withSomething:world:test -indexed-file=%s -indexed-at=67:18 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK11 %s
// RUN: clang-refactor-test rename-indexed-file -name=z_Z_42:onEntity:a_200:onEntity -new-name=part:world:z_Z_42:onEntity -indexed-file=%s -indexed-at=73:18 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK12 %s

@implementation method
-(BOOL)usingThing:(void)part part:(void)world
class:(int)part perform:(some_type_t)withSomething
z_Z_42:(const Object &)class //comment
usingThing:(({}))part {
  some_type_t foo = [self.undef_property onEntity: "string" withSomething: ']'];
}
// CHECK13: [[@LINE-6]]:8 -> [[@LINE-6]]:18, [[@LINE-6]]:30 -> [[@LINE-6]]:34, [[@LINE-5]]:1 -> [[@LINE-5]]:6, [[@LINE-5]]:17 -> [[@LINE-5]]:24, [[@LINE-4]]:1 -> [[@LINE-4]]:7, [[@LINE-3]]:1 -> [[@LINE-3]]:11
@end
// RUN: clang-refactor-test rename-indexed-file -name=usingThing:part:class:perform:z_Z_42:usingThing -new-name=class:method:withSomething:world:name:name -indexed-file=%s -indexed-at=87:8 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK13 %s

@interface withSomething
-(Object *)object:(^ { })onEntity __attribute__((test()))piece:(^ { })z_Z_42 /*comment*/ ;
// CHECK14: [[@LINE-1]]:12 -> [[@LINE-1]]:18, [[@LINE-1]]:58 -> [[@LINE-1]]:63
@end
// RUN: clang-refactor-test rename-indexed-file -name=object:piece -new-name=foo:bar -indexed-file=%s -indexed-at=98:12 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK14 %s

@interface usingThing <piece, withSomething, perform>
+(void)part:(some_type_t)onEntity
a_200:(int)bar perform:(some_type_t)struct
;
// CHECK15: [[@LINE-3]]:8 -> [[@LINE-3]]:12, [[@LINE-2]]:1 -> [[@LINE-2]]:6, [[@LINE-2]]:16 -> [[@LINE-2]]:23
-(some_type_t)piece:(^ { })foo name:(({}))perform part:(^ { })name z_Z_42:(Object *)test
;
// CHECK16: [[@LINE-2]]:15 -> [[@LINE-2]]:20, [[@LINE-2]]:32 -> [[@LINE-2]]:36, [[@LINE-2]]:51 -> [[@LINE-2]]:55, [[@LINE-2]]:68 -> [[@LINE-2]]:74
+(Object *)method:(const Object &)method onEntity:(^ { })withSomething
withSomething:(int (*)(const Object &, BOOL))piece ;
// CHECK17: [[@LINE-2]]:12 -> [[@LINE-2]]:18, [[@LINE-2]]:42 -> [[@LINE-2]]:50, [[@LINE-1]]:1 -> [[@LINE-1]]:14
-(BOOL)foo:(int[1 + 2 - 3])world
foo:(int[1 + 2 - 3])struct method:(void (*)())piece foo:(BOOL)test __attribute__((eval { int x = 0 + 1; })) ;
// CHECK18: [[@LINE-2]]:8 -> [[@LINE-2]]:11, [[@LINE-1]]:1 -> [[@LINE-1]]:4, [[@LINE-1]]:28 -> [[@LINE-1]]:34, [[@LINE-1]]:53 -> [[@LINE-1]]:56
@end
// RUN: clang-refactor-test rename-indexed-file -name=part:a_200:perform -new-name=part:test:part -indexed-file=%s -indexed-at=104:8 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK15 %s
// RUN: clang-refactor-test rename-indexed-file -name=piece:name:part:z_Z_42 -new-name=usingThing:struct:onEntity:piece -indexed-file=%s -indexed-at=108:15 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK16 %s
// RUN: clang-refactor-test rename-indexed-file -name=method:onEntity:withSomething -new-name=z_Z_42:method:usingThing -indexed-file=%s -indexed-at=111:12 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK17 %s
// RUN: clang-refactor-test rename-indexed-file -name=foo:foo:method:foo -new-name=piece:usingThing:class:foo -indexed-file=%s -indexed-at=114:8 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK18 %s

@interface foo
-(void)foo:(({}))part method:(({}))method ;
// CHECK19: [[@LINE-1]]:8 -> [[@LINE-1]]:11, [[@LINE-1]]:23 -> [[@LINE-1]]:29
-(const Object &)foo:(({}))usingThing test:(Object *)struct __attribute__((test()))name:(const Object &)z_Z_42 /*comment*/ name:(^ { })onEntity a_200:(Object * (^)())usingThing
usingThing:(Object *)method /*comment*/ ;
// CHECK20: [[@LINE-2]]:18 -> [[@LINE-2]]:21, [[@LINE-2]]:39 -> [[@LINE-2]]:43, [[@LINE-2]]:84 -> [[@LINE-2]]:88, [[@LINE-2]]:124 -> [[@LINE-2]]:128, [[@LINE-2]]:145 -> [[@LINE-2]]:150, [[@LINE-1]]:1 -> [[@LINE-1]]:11
+(BOOL)name:(Object *)a_200 part:(int)bar
perform:(some_type_t)method ;
// CHECK21: [[@LINE-2]]:8 -> [[@LINE-2]]:12, [[@LINE-2]]:29 -> [[@LINE-2]]:33, [[@LINE-1]]:1 -> [[@LINE-1]]:8
-(some_type_t)usingThing:(void)test //comment
foo:(void)world z_Z_42:(Object *)bar ;
// CHECK22: [[@LINE-2]]:15 -> [[@LINE-2]]:25, [[@LINE-1]]:1 -> [[@LINE-1]]:4, [[@LINE-1]]:17 -> [[@LINE-1]]:23
@end
// RUN: clang-refactor-test rename-indexed-file -name=foo:method -new-name=part:world -indexed-file=%s -indexed-at=124:8 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK19 %s
// RUN: clang-refactor-test rename-indexed-file -name=foo:test:name:name:a_200:usingThing -new-name=method:withSomething:struct:z_Z_42:bar:z_Z_42 -indexed-file=%s -indexed-at=126:18 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK20 %s
// RUN: clang-refactor-test rename-indexed-file -name=name:part:perform -new-name=withSomething:part:object -indexed-file=%s -indexed-at=129:8 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK21 %s
// RUN: clang-refactor-test rename-indexed-file -name=usingThing:foo:z_Z_42 -new-name=perform:test:class -indexed-file=%s -indexed-at=132:15 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK22 %s

@implementation struct
+(void)bar:(void)piece a_200:(some_type_t)class {
  [self name: [] () {
   [globalObject send: [super name: [] () {


 } perform: globalArray[i]] other: 42];

 } usingThing: @"string literal"];

  call() = [self usingThing: @"string literal" perform: "string"];
}
// CHECK23: [[@LINE-11]]:8 -> [[@LINE-11]]:11, [[@LINE-11]]:24 -> [[@LINE-11]]:29
-(void)perform:(BOOL)test
withSomething:(const Object &)foo {
  return [_undef_ivar piece: globalArray[i] + globalArray[i] object: globalArray[i]
 a_200: [self withSomething: [] () {    [self.undef_property usingThing: ']' struct: "]"
 perform: "string"];
  }
 piece: ']' name: ']' a_200: "string" == ^ () {
   globalArray[12] = [_undef_ivar foo: ']' foo: []  {
  }
 bar: ^  {
  } + []  {


 }
];

 }] z_Z_42: ^ () {
   if (^ () {
  }) {
      return @"string literal" < globalArray[i] == [self world: []  {


 } object: @"string literal" part: ']'];

  }

 } a_200: "string"];
}
// CHECK24: [[@LINE-28]]:8 -> [[@LINE-28]]:15, [[@LINE-27]]:1 -> [[@LINE-27]]:14
@end
// RUN: clang-refactor-test rename-indexed-file -name=bar:a_200 -new-name=bar:name -indexed-file=%s -indexed-at=142:8 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK23 %s
// RUN: clang-refactor-test rename-indexed-file -name=perform:withSomething -new-name=withSomething:test -indexed-file=%s -indexed-at=154:8 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK24 %s

@implementation object
-(int)a_200:(Object *)perform usingThing:(int)bar //comment
{
  [self bar: ']' withSomething: "string"];

  Object * part = ']';
}
// CHECK25: [[@LINE-6]]:7 -> [[@LINE-6]]:12, [[@LINE-6]]:31 -> [[@LINE-6]]:41
@end
// RUN: clang-refactor-test rename-indexed-file -name=a_200:usingThing -new-name=object:bar -indexed-file=%s -indexed-at=188:7 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK25 %s

@implementation test
+(void)method:(const Object &)z_Z_42 //comment
piece:(some_type_t (*)(Object *))withSomething __attribute__((eval { int x = 0 + 1; })) perform:(void)piece method:(BOOL)object {
  [self piece: [] () {
   ; ;[self part: 12 a_200: @"string literal"];

 } struct: globalArray[i] part: "]"];

  ;
}
// CHECK26: [[@LINE-9]]:8 -> [[@LINE-9]]:14, [[@LINE-8]]:1 -> [[@LINE-8]]:6, [[@LINE-8]]:89 -> [[@LINE-8]]:96, [[@LINE-8]]:109 -> [[@LINE-8]]:115
-(some_type_t)world:(some_type_t)world
a_200:(void)bar //comment
{
  int piece = ;

  if ("string") {
      // comment

  }
}
// CHECK27: [[@LINE-10]]:15 -> [[@LINE-10]]:20, [[@LINE-9]]:1 -> [[@LINE-9]]:6
@end
// RUN: clang-refactor-test rename-indexed-file -name=method:piece:perform:method -new-name=perform:test:bar:object -indexed-file=%s -indexed-at=199:8 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK26 %s
// RUN: clang-refactor-test rename-indexed-file -name=world:a_200 -new-name=z_Z_42:method -indexed-file=%s -indexed-at=209:15 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK27 %s

@implementation perform
-(Object *)class:(Object * (*)(BOOL))onEntity perform:(({}))class
usingThing:(int (*)())object __attribute__((eval { int x = 0 + 1; })) piece:(BOOL (^)(BOOL, BOOL))class bar:(int)name withSomething:(const Object &)onEntity __attribute__((test())){
  // comment
}
// CHECK28: [[@LINE-4]]:12 -> [[@LINE-4]]:17, [[@LINE-4]]:47 -> [[@LINE-4]]:54, [[@LINE-3]]:1 -> [[@LINE-3]]:11, [[@LINE-3]]:71 -> [[@LINE-3]]:76, [[@LINE-3]]:105 -> [[@LINE-3]]:108, [[@LINE-3]]:119 -> [[@LINE-3]]:132
-(int)test:(({}))onEntity
a_200:(BOOL)struct {
  ([self name: ']' part: []  {
   ; ;([_undef_ivar usingThing: globalArray[i] class: globalArray[i] method: globalArray[i] method: ']' class: 12]);

 } bar: 12 withSomething: "]" == []  {    [self test: [self world: [_undef_ivar struct: "]" bar: @"string literal" method: globalArray[i]] usingThing: @"string literal" class: [] () {
  } bar: @"string literal"
] withSomething: ([]  {


 } + "]" * ']') withSomething: @"string literal"];
  } usingThing: "]"]);

  call() = [self test: globalArray[i] name: @"string literal" perform: [self class: "]"
 piece: globalArray[i] method: "string"
]
//comment
 piece: "string"];
}
// CHECK29: [[@LINE-19]]:7 -> [[@LINE-19]]:11, [[@LINE-18]]:1 -> [[@LINE-18]]:6
-(Object *)test:(^ { })z_Z_42 class:(void)usingThing
{
  if ([self.undef_property class: globalArray[i] a_200: ^ () { ] }
]) {
      if (globalArray[i]) {
      some_type_t foo = [self foo: "]" perform: "string"
];

  }

  }
}
// CHECK30: [[@LINE-12]]:12 -> [[@LINE-12]]:16, [[@LINE-12]]:31 -> [[@LINE-12]]:36
+(int)struct:(void (^)())method
a_200:(BOOL)onEntity usingThing:(Object *)a_200 bar:(void)z_Z_42 z_Z_42:(Object *)struct perform:(BOOL (*)(Object *, Object *))method {
  globalArray[12] = [self bar: ']' test: ^ () { ] } method: globalArray[i]];

  if ([super test: "string" < globalArray[i] part: 12]) {
      int bar = [self object: "]"
 perform: [self a_200: "string" object: ("string")
//comment
]];

  }
}
// CHECK31: [[@LINE-12]]:7 -> [[@LINE-12]]:13, [[@LINE-11]]:1 -> [[@LINE-11]]:6, [[@LINE-11]]:22 -> [[@LINE-11]]:32, [[@LINE-11]]:49 -> [[@LINE-11]]:52, [[@LINE-11]]:66 -> [[@LINE-11]]:72, [[@LINE-11]]:90 -> [[@LINE-11]]:97
@end
// RUN: clang-refactor-test rename-indexed-file -name=class:perform:usingThing:piece:bar:withSomething -new-name=a_200:piece:foo:struct:onEntity:world -indexed-file=%s -indexed-at=225:12 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK28 %s
// RUN: clang-refactor-test rename-indexed-file -name=test:a_200 -new-name=struct:world -indexed-file=%s -indexed-at=230:7 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK29 %s
// RUN: clang-refactor-test rename-indexed-file -name=test:class -new-name=perform:name -indexed-file=%s -indexed-at=250:12 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK30 %s
// RUN: clang-refactor-test rename-indexed-file -name=struct:a_200:usingThing:bar:z_Z_42:perform -new-name=world:piece:test:world:class:struct -indexed-file=%s -indexed-at=263:7 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK31 %s

@implementation usingThing <object>
-(int)foo:(int)a_200 part:(Object *)name
object:(BOOL)world __attribute__((test()))method:(const Object & (*)())test name:(some_type_t)z_Z_42 bar:(Object *)class /*comment*/ {
  [self piece: ^  {
   [self.undef_property name: [self z_Z_42: [self object: @"string literal" + []  {
  }
 z_Z_42: [] () {


 }] class: [self method: globalArray[i] usingThing: globalArray[i] part: 12] onEntity: @{ @1, @3 } piece: "string"] object: ([self foo: @"string literal" < [self object: ']' usingThing:  usingThing: globalArray[i] a_200: [] () {
  }]
 name: "string" foo: [] () {


 }]) struct: "string" struct: "]" + []  {
  } usingThing: ];

 } usingThing: (@"string literal") usingThing: 12 a_200: ^  {    [globalObject message] = [self onEntity: []  {
  } == ^  {


 } foo: @{ @1, @3 }];
  } withSomething: "]" == ];
}
// CHECK32: [[@LINE-23]]:7 -> [[@LINE-23]]:10, [[@LINE-23]]:22 -> [[@LINE-23]]:26, [[@LINE-22]]:1 -> [[@LINE-22]]:7, [[@LINE-22]]:43 -> [[@LINE-22]]:49, [[@LINE-22]]:77 -> [[@LINE-22]]:81, [[@LINE-22]]:102 -> [[@LINE-22]]:105
-(Object *)withSomething:(void)class __attribute__((test()))onEntity:(void)part struct:(some_type_t (*)(Object *))bar __attribute__((eval { int x = 0 + 1; })) perform:(Object *)world {
  globalArray[12] = [self struct: globalArray[i] bar: ^  {    if (']' + "string") {
      return ^ () {
  };

  }
  } class: ^ () {    some_type_t foo = [self.undef_property withSomething: ']' bar: "]"];
  } usingThing: ']'
];
}
// CHECK33: [[@LINE-10]]:12 -> [[@LINE-10]]:25, [[@LINE-10]]:61 -> [[@LINE-10]]:69, [[@LINE-10]]:81 -> [[@LINE-10]]:87, [[@LINE-10]]:160 -> [[@LINE-10]]:167
@end
// RUN: clang-refactor-test rename-indexed-file -name=foo:part:object:method:name:bar -new-name=z_Z_42:z_Z_42:part:part:usingThing:z_Z_42 -indexed-file=%s -indexed-at=283:7 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK32 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:onEntity:struct:perform -new-name=withSomething:name:foo:name -indexed-file=%s -indexed-at=307:12 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK33 %s

@interface a_200 <foo>
+(void)bar:(int[1 + 2 - 3])a_200 object:(Object *)z_Z_42 world:(int[1 + 2 - 3])name name:(const Object &)name world:(Object *)world /*comment*/ ;
// CHECK34: [[@LINE-1]]:8 -> [[@LINE-1]]:11, [[@LINE-1]]:34 -> [[@LINE-1]]:40, [[@LINE-1]]:58 -> [[@LINE-1]]:63, [[@LINE-1]]:85 -> [[@LINE-1]]:89, [[@LINE-1]]:111 -> [[@LINE-1]]:116
-(int)foo:(void (^)())part usingThing:(int)usingThing object:(({}))withSomething class:(some_type_t (^)())perform /*comment*/ method:(int)foo ;
// CHECK35: [[@LINE-1]]:7 -> [[@LINE-1]]:10, [[@LINE-1]]:28 -> [[@LINE-1]]:38, [[@LINE-1]]:55 -> [[@LINE-1]]:61, [[@LINE-1]]:82 -> [[@LINE-1]]:87, [[@LINE-1]]:127 -> [[@LINE-1]]:133
-(void)z_Z_42:(({}))withSomething name:(void (*)())struct foo:(Object *)part ;
// CHECK36: [[@LINE-1]]:8 -> [[@LINE-1]]:14, [[@LINE-1]]:35 -> [[@LINE-1]]:39, [[@LINE-1]]:59 -> [[@LINE-1]]:62
-(BOOL)onEntity:(const Object &)z_Z_42
piece:(const Object &)world
onEntity:(BOOL)part ;
// CHECK37: [[@LINE-3]]:8 -> [[@LINE-3]]:16, [[@LINE-2]]:1 -> [[@LINE-2]]:6, [[@LINE-1]]:1 -> [[@LINE-1]]:9
-(BOOL)method:(some_type_t)name
part:(BOOL)a_200 part:(void)test __attribute__((eval { int x = 0 + 1; })) ;
// CHECK38: [[@LINE-2]]:8 -> [[@LINE-2]]:14, [[@LINE-1]]:1 -> [[@LINE-1]]:5, [[@LINE-1]]:18 -> [[@LINE-1]]:22
@end
// RUN: clang-refactor-test rename-indexed-file -name=bar:object:world:name:world -new-name=withSomething:part:struct:part:withSomething -indexed-file=%s -indexed-at=323:8 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK34 %s
// RUN: clang-refactor-test rename-indexed-file -name=foo:usingThing:object:class:method -new-name=part:onEntity:foo:test:struct -indexed-file=%s -indexed-at=325:7 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK35 %s
// RUN: clang-refactor-test rename-indexed-file -name=z_Z_42:name:foo -new-name=piece:z_Z_42:world -indexed-file=%s -indexed-at=327:8 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK36 %s
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:piece:onEntity -new-name=part:usingThing:foo -indexed-file=%s -indexed-at=329:8 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK37 %s
// RUN: clang-refactor-test rename-indexed-file -name=method:part:part -new-name=perform:test:part -indexed-file=%s -indexed-at=333:8 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK38 %s

@interface part <world, struct, part>
+(some_type_t)z_Z_42:(Object *)onEntity test:(int (*)())a_200 __attribute__((test()));
// CHECK39: [[@LINE-1]]:15 -> [[@LINE-1]]:21, [[@LINE-1]]:41 -> [[@LINE-1]]:45
@end
// RUN: clang-refactor-test rename-indexed-file -name=z_Z_42:test -new-name=name:onEntity -indexed-file=%s -indexed-at=344:15 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK39 %s

@interface piece
-(some_type_t)part:(some_type_t)onEntity test:(Object *)world
bar:(BOOL)foo
method:(const Object & (^)(int, some_type_t))object
withSomething:(BOOL)part
;
// CHECK40: [[@LINE-5]]:15 -> [[@LINE-5]]:19, [[@LINE-5]]:42 -> [[@LINE-5]]:46, [[@LINE-4]]:1 -> [[@LINE-4]]:4, [[@LINE-3]]:1 -> [[@LINE-3]]:7, [[@LINE-2]]:1 -> [[@LINE-2]]:14
-(some_type_t)onEntity:(void)object test:(int[1 + 2 - 3])object ;
// CHECK41: [[@LINE-1]]:15 -> [[@LINE-1]]:23, [[@LINE-1]]:37 -> [[@LINE-1]]:41
-(some_type_t)struct:(void)name class:(some_type_t)foo name:(int (^)(BOOL, int))method
foo:(void)usingThing usingThing:(int)z_Z_42 /*comment*/ a_200:(int[1 + 2 - 3])object
;
// CHECK42: [[@LINE-3]]:15 -> [[@LINE-3]]:21, [[@LINE-3]]:33 -> [[@LINE-3]]:38, [[@LINE-3]]:56 -> [[@LINE-3]]:60, [[@LINE-2]]:1 -> [[@LINE-2]]:4, [[@LINE-2]]:22 -> [[@LINE-2]]:32, [[@LINE-2]]:57 -> [[@LINE-2]]:62
-(BOOL)bar:(some_type_t (*)(int))part /*comment*/ object:(int[1 + 2 - 3])onEntity perform:(BOOL)usingThing /*comment*/ ;
// CHECK43: [[@LINE-1]]:8 -> [[@LINE-1]]:11, [[@LINE-1]]:51 -> [[@LINE-1]]:57, [[@LINE-1]]:83 -> [[@LINE-1]]:90
@end
// RUN: clang-refactor-test rename-indexed-file -name=part:test:bar:method:withSomething -new-name=name:usingThing:perform:onEntity:bar -indexed-file=%s -indexed-at=350:15 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK40 %s
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:test -new-name=onEntity:class -indexed-file=%s -indexed-at=356:15 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK41 %s
// RUN: clang-refactor-test rename-indexed-file -name=struct:class:name:foo:usingThing:a_200 -new-name=class:a_200:class:object:class:name -indexed-file=%s -indexed-at=358:15 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK42 %s
// RUN: clang-refactor-test rename-indexed-file -name=bar:object:perform -new-name=a_200:test:object -indexed-file=%s -indexed-at=362:8 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK43 %s

@implementation piece
+(BOOL)name:(int (^)(BOOL, const Object &))onEntity /*comment*/ withSomething:(void)foo name:(const Object &)withSomething a_200:(Object *)struct foo:(^ { })object {
  if (12) {
      return 12;

  }
}
// CHECK44: [[@LINE-6]]:8 -> [[@LINE-6]]:12, [[@LINE-6]]:65 -> [[@LINE-6]]:78, [[@LINE-6]]:89 -> [[@LINE-6]]:93, [[@LINE-6]]:124 -> [[@LINE-6]]:129, [[@LINE-6]]:147 -> [[@LINE-6]]:150
+(int)class:(some_type_t (*)(BOOL))name
name:(BOOL)onEntity __attribute__((test()))a_200:(^ { })withSomething __attribute__((eval { int x = 0 + 1; })) piece:(int)withSomething usingThing:(Object *)name {
  ; ;[super method:  part: ^ () {
   // comment

 }];

  [self struct: "string" == "]" withSomething: ']' class: [] () {
   [globalObject send: [self.undef_property part: ^ () {


 } class: (']' < )
 z_Z_42: ^ () {


 } test: (12)] other: 42];

 } object: 12 perform: ']'];
}
// CHECK45: [[@LINE-19]]:7 -> [[@LINE-19]]:12, [[@LINE-18]]:1 -> [[@LINE-18]]:5, [[@LINE-18]]:44 -> [[@LINE-18]]:49, [[@LINE-18]]:112 -> [[@LINE-18]]:117, [[@LINE-18]]:137 -> [[@LINE-18]]:147
+(some_type_t)perform:(int[1 + 2 - 3])world //comment
withSomething:(const Object &)foo part:(int (^)(int))foo part:(int[1 + 2 - 3])perform //comment
{
  [globalObject send: [self object: ']' withSomething: 12 usingThing: ']' perform: @"string literal"] other: 42];
}
// CHECK46: [[@LINE-5]]:15 -> [[@LINE-5]]:22, [[@LINE-4]]:1 -> [[@LINE-4]]:14, [[@LINE-4]]:35 -> [[@LINE-4]]:39, [[@LINE-4]]:58 -> [[@LINE-4]]:62
@end
// RUN: clang-refactor-test rename-indexed-file -name=name:withSomething:name:a_200:foo -new-name=object:onEntity:world:world:piece -indexed-file=%s -indexed-at=371:8 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK44 %s
// RUN: clang-refactor-test rename-indexed-file -name=class:name:a_200:piece:usingThing -new-name=method:test:z_Z_42:part:foo -indexed-file=%s -indexed-at=378:7 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK45 %s
// RUN: clang-refactor-test rename-indexed-file -name=perform:withSomething:part:part -new-name=a_200:object:perform:z_Z_42 -indexed-file=%s -indexed-at=398:15 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK46 %s

@implementation z_Z_42 <struct>
+(BOOL)piece:(some_type_t (^)(int))class a_200:(BOOL (^)(some_type_t, BOOL))bar
world:(const Object & (*)(const Object &, const Object &))struct piece:(^ { })test z_Z_42:(BOOL)name
usingThing:(void (^)(Object *))struct {
  [self object: ([self.undef_property struct: ^  {
   /*comment*/[self.undef_property part: @"string literal" world: @"string literal"
 object: "]" world: "string" world: @{ @1, @3 }];

 } == ^ () {    [super class: ^ () {


 } * globalArray[i] piece: []  {
  } bar: [self part: @"string literal" perform: "]" world: [] () {
  }
 world: [self.undef_property world: globalArray[i] perform: "string" struct: "]"]]];
  }
 name: 12 usingThing: "]" bar: "]"
//comment
]) withSomething: ^ () {
   if ("string") {
      [self withSomething: []  {


 } method: 12 piece: [] () {


 } a_200: ']' method: "]"];

  }

 } test: @{ @1, @3 }
 class: 12 world: "string"];
}
// CHECK47: [[@LINE-32]]:8 -> [[@LINE-32]]:13, [[@LINE-32]]:42 -> [[@LINE-32]]:47, [[@LINE-31]]:1 -> [[@LINE-31]]:6, [[@LINE-31]]:66 -> [[@LINE-31]]:71, [[@LINE-31]]:84 -> [[@LINE-31]]:90, [[@LINE-30]]:1 -> [[@LINE-30]]:11
-(int)method:(Object *)bar //comment
part:(int)bar usingThing:(({}))name {
  // comment

  [_undef_ivar world: "]" onEntity: []  {
   ; ;[self part: globalArray[i] world: @{ @1, @3 }];

 } bar: "string" == [super name: 12 piece: /*]*/ @"string literal" part: "string" method: ^  {
   some_type_t object = "string";

 } piece: (^  {
   return /*]*/ "]";

 })] onEntity: []  {
   int bar = [self name: "string"/*comment*/ method: globalArray[i]];

 }];
}
// CHECK48: [[@LINE-18]]:7 -> [[@LINE-18]]:13, [[@LINE-17]]:1 -> [[@LINE-17]]:5, [[@LINE-17]]:15 -> [[@LINE-17]]:25
+(Object *)class:(BOOL)object __attribute__((test()))method:(some_type_t)method
world:(Object *)struct
{
  [self method: ^  {    call() = [self a_200: ']' a_200: "]"];
  } struct: ^ () {
   return [] () {
  };

 } class: [_undef_ivar part: "string" name: []  {
   int bar = [self withSomething: "string" bar: @"string literal" usingThing: @"string literal"];

 }]];

  [super foo: 12 foo:  * "]" foo: ("string") foo: "string" method: 12];
}
// CHECK49: [[@LINE-15]]:12 -> [[@LINE-15]]:17, [[@LINE-15]]:54 -> [[@LINE-15]]:60, [[@LINE-14]]:1 -> [[@LINE-14]]:6
@end
// RUN: clang-refactor-test rename-indexed-file -name=piece:a_200:world:piece:z_Z_42:usingThing -new-name=struct:part:bar:struct:class:usingThing -indexed-file=%s -indexed-at=410:8 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK47 %s
// RUN: clang-refactor-test rename-indexed-file -name=method:part:usingThing -new-name=object:piece:method -indexed-file=%s -indexed-at=443:7 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK48 %s
// RUN: clang-refactor-test rename-indexed-file -name=class:method:world -new-name=struct:test:z_Z_42 -indexed-file=%s -indexed-at=462:12 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK49 %s

@implementation a_200 <name, bar, a_200>
+(some_type_t)class:(void)z_Z_42
object:(BOOL)perform
z_Z_42:(const Object &)usingThing name:(const Object &)a_200 test:(({}))world perform:(int[1 + 2 - 3])z_Z_42
{
  if ("string") {
      if ([super withSomething: []  {    return [_undef_ivar usingThing: [] () {
  }
 world: [] () {


 } foo: 12 perform: globalArray[i] name: @"string literal"];
  } world: ^  {    // comment
  } object: ^  {    int method = "]";
  }
]) {
      [globalObject send: [self.undef_property method: globalArray[i] object: []  {    [self method: ']' test: [self foo: ("string") a_200: @"string literal" piece: 12 < [self class: @"string literal" piece: "]" + ']']] test: [self onEntity: @"string literal"
 z_Z_42: ']'] name: globalArray[i] test: 12] other: 42];
  } class: ^  {    ; ;[self perform: ("string") bar: (']') foo: "string"
 foo: ^  {


 }];
  } piece: ^  {
   call() = [_undef_ivar perform: (12) bar: ^  {


 } test: "]" name: "string" bar: "]"];

 }];

  }

  }

  [globalObject send: [self bar: globalArray[i] == ']' perform: [_undef_ivar z_Z_42: @"string literal" object: ^  {
   [globalObject send: [self world: "]" onEntity: ']'
 struct: [] () {
  } perform: [self.undef_property piece: [] () {


 } method: 12 foo: []  {


 } onEntity: "string" * "]" method: "]"] < ']'] other: 42] other: 42];

 } method: ^  {    // comment
  } test: ^ () {    return []  {
  };
  }] a_200: [] () {    ([self.undef_property object: ^  {
  } usingThing: @"string literal"
 test: "string" usingThing: globalArray[i]
 foo: /*]*/ globalArray[i]]);
  } piece: 12];
}
// CHECK50: [[@LINE-54]]:15 -> [[@LINE-54]]:20, [[@LINE-53]]:1 -> [[@LINE-53]]:7, [[@LINE-52]]:1 -> [[@LINE-52]]:7, [[@LINE-52]]:35 -> [[@LINE-52]]:39, [[@LINE-52]]:62 -> [[@LINE-52]]:66, [[@LINE-52]]:79 -> [[@LINE-52]]:86
@end
// RUN: clang-refactor-test rename-indexed-file -name=class:object:z_Z_42:name:test:perform -new-name=world:foo:part:a_200:name:piece -indexed-file=%s -indexed-at=484:15 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK50 %s

@implementation piece (part)
+(BOOL)method:(^ { })foo foo:(({}))foo
usingThing:(int)withSomething perform:(int[1 + 2 - 3])bar a_200:(some_type_t (^)())onEntity test:(const Object &)z_Z_42 __attribute__((eval { int x = 0 + 1; })) {
  if ([self method: ']' struct: ^  {    [globalObject send: [_undef_ivar world: "]" onEntity: globalArray[i] foo: "]" withSomething:  < ']' onEntity: "string"] other: 42];
  }
]) {
      call() = [self part: 12
 struct: "string" perform: "string" class: [] () {    return "string";
  }
];

  }
}
// CHECK51: [[@LINE-12]]:8 -> [[@LINE-12]]:14, [[@LINE-12]]:26 -> [[@LINE-12]]:29, [[@LINE-11]]:1 -> [[@LINE-11]]:11, [[@LINE-11]]:31 -> [[@LINE-11]]:38, [[@LINE-11]]:59 -> [[@LINE-11]]:64, [[@LINE-11]]:93 -> [[@LINE-11]]:97
+(some_type_t)withSomething:(some_type_t (^)(int, Object *))z_Z_42 bar:(^ { })usingThing {
  return 12;
}
// CHECK52: [[@LINE-3]]:15 -> [[@LINE-3]]:28, [[@LINE-3]]:68 -> [[@LINE-3]]:71
-(some_type_t)object:(some_type_t (*)())bar
foo:(some_type_t (*)(const Object &))class
withSomething:(void)onEntity
usingThing:(^ { })bar a_200:(some_type_t (^)())usingThing
usingThing:(^ { })name {
  call() = ([self part: "]" piece: "string" * "]" bar: globalArray[i] piece: [] () {    int bar = [self test: "string" method: globalArray[i] method: "string" == /*]*/ "string" * @"string literal" object: @"string literal" usingThing: globalArray[i]];
  } < "string"]);

  if ("]" + [self part: @"string literal" class: "string"]) {
      [super onEntity: "]" object: "]" piece: (12) test: ']' < ^  {
   int bar = [self usingThing: [] () {


 } z_Z_42: ^  {
  } usingThing: "]" object: "]"];

 }
];

  }
}
// CHECK53: [[@LINE-21]]:15 -> [[@LINE-21]]:21, [[@LINE-20]]:1 -> [[@LINE-20]]:4, [[@LINE-19]]:1 -> [[@LINE-19]]:14, [[@LINE-18]]:1 -> [[@LINE-18]]:11, [[@LINE-18]]:23 -> [[@LINE-18]]:28, [[@LINE-17]]:1 -> [[@LINE-17]]:11
@end
// RUN: clang-refactor-test rename-indexed-file -name=method:foo:usingThing:perform:a_200:test -new-name=object:z_Z_42:test:struct:perform:z_Z_42 -indexed-file=%s -indexed-at=543:8 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK51 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:bar -new-name=world:foo -indexed-file=%s -indexed-at=556:15 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK52 %s
// RUN: clang-refactor-test rename-indexed-file -name=object:foo:withSomething:usingThing:a_200:usingThing -new-name=part:a_200:method:perform:world:part -indexed-file=%s -indexed-at=560:15 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK53 %s

@interface world
-(const Object &)class:(int[1 + 2 - 3])struct onEntity:(void (^)(Object *, int))withSomething __attribute__((test()));
// CHECK54: [[@LINE-1]]:18 -> [[@LINE-1]]:23, [[@LINE-1]]:47 -> [[@LINE-1]]:55
-(int)world:(int)foo name:(const Object & (^)(int, Object *))piece a_200:(Object *)test object:(BOOL)onEntity __attribute__((eval { int x = 0 + 1; })) onEntity:(BOOL)z_Z_42 /*comment*/ ;
// CHECK55: [[@LINE-1]]:7 -> [[@LINE-1]]:12, [[@LINE-1]]:22 -> [[@LINE-1]]:26, [[@LINE-1]]:68 -> [[@LINE-1]]:73, [[@LINE-1]]:89 -> [[@LINE-1]]:95, [[@LINE-1]]:152 -> [[@LINE-1]]:160
-(int)usingThing:(^ { })part name:(const Object &)usingThing usingThing:(^ { })perform __attribute__((test()))struct:(^ { })world
withSomething:(int[1 + 2 - 3])method //comment
;
// CHECK56: [[@LINE-3]]:7 -> [[@LINE-3]]:17, [[@LINE-3]]:30 -> [[@LINE-3]]:34, [[@LINE-3]]:62 -> [[@LINE-3]]:72, [[@LINE-3]]:111 -> [[@LINE-3]]:117, [[@LINE-2]]:1 -> [[@LINE-2]]:14
@end
// RUN: clang-refactor-test rename-indexed-file -name=class:onEntity -new-name=usingThing:usingThing -indexed-file=%s -indexed-at=588:18 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK54 %s
// RUN: clang-refactor-test rename-indexed-file -name=world:name:a_200:object:onEntity -new-name=world:bar:onEntity:name:part -indexed-file=%s -indexed-at=590:7 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK55 %s
// RUN: clang-refactor-test rename-indexed-file -name=usingThing:name:usingThing:struct:withSomething -new-name=name:struct:method:method:usingThing -indexed-file=%s -indexed-at=592:7 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK56 %s

@implementation onEntity
-(BOOL)piece:(some_type_t (^)(int))usingThing perform:(int)foo onEntity:(({}))withSomething __attribute__((test()))bar:(Object *)usingThing usingThing:(const Object &)bar class:(const Object & (^)(int, const Object &))perform {
  [self object: "string" foo: ([self onEntity: "string" * []  {    if (@"string literal") {
      call() = [self z_Z_42: "string" usingThing: @"string literal"
];

  }
  } object: globalArray[i]
 test: ^ () {    if ("string" == @"string literal" + ^ () {


 }) {
      int name = 12 * 12;

  }
  }]) foo: @{ @1, @3 }];

  call() = [super foo: ^ () { ] } part: ']' onEntity: [self piece: ']' piece: [] () {    [super name: "]" perform: "string"];
  }]
 struct: "]" part: @"string literal"];
}
// CHECK57: [[@LINE-20]]:8 -> [[@LINE-20]]:13, [[@LINE-20]]:47 -> [[@LINE-20]]:54, [[@LINE-20]]:64 -> [[@LINE-20]]:72, [[@LINE-20]]:116 -> [[@LINE-20]]:119, [[@LINE-20]]:141 -> [[@LINE-20]]:151, [[@LINE-20]]:172 -> [[@LINE-20]]:177
-(int)withSomething:(int[1 + 2 - 3])foo usingThing:(Object * (^)(some_type_t, BOOL))a_200 perform:(some_type_t)onEntity __attribute__((test())){
  [globalObject message] = [self.undef_property struct: @"string literal" object: "]" < ']' withSomething: @"string literal" piece: [self.undef_property withSomething: 12 foo: @"string literal" withSomething: "]" object: [] () {    Object * z_Z_42 = 12 + "string";
  } part: ^ () {
   const Object & z_Z_42 = globalArray[i];

 }]];

  some_type_t foo = [self.undef_property foo: 12 * globalArray[i] method: @"string literal" a_200: ^ () {
   globalArray[12] = [self onEntity: @"string literal" class: "string" onEntity: "]"];

 }/*comment*/];
}
// CHECK58: [[@LINE-12]]:7 -> [[@LINE-12]]:20, [[@LINE-12]]:41 -> [[@LINE-12]]:51, [[@LINE-12]]:91 -> [[@LINE-12]]:98
+(int)piece:(int)test
onEntity:(int[1 + 2 - 3])test piece:(const Object &)piece {
  [self z_Z_42: ']' usingThing: ("]")];

  if ([self test: [] () {
   // comment

 } a_200: [] () {    BOOL foo = [self world: [self.undef_property test: globalArray[i] foo: [self method: "]" onEntity: []  {


 } withSomething: globalArray[i]] piece: 12 struct: ']'] usingThing: [super test: globalArray[i] world: [_undef_ivar bar: @{ @1, @3 } z_Z_42: [_undef_ivar bar:  z_Z_42: [self foo: ("string") object: ^ () {


 } bar: "string"] struct: (12) withSomething: @"string literal" name: [self.undef_property part: @"string literal" onEntity: []  {


 }]] onEntity: ']' name: 12 object: @{ @1, @3 }]
//comment
] part: globalArray[i] test: 12];
  } + ']' bar: [super piece: "]" world: @"string literal" object: globalArray[i] test: ']'
 withSomething: @{ @1, @3 }] piece: [self a_200: ']' world: ']'] onEntity: @"string literal"] == ^ () { ] }) {
      if (12) {
      [self.undef_property withSomething: [self a_200: globalArray[i] withSomething: "string"
 test: [] () {    [self perform: "]" object: ']' foo: @{ @1, @3 } + [] () {
  }];
  } object: "string" struct: /*]*/ "]"] piece: @{ @1, @3 } a_200: 12 bar: globalArray[i]];

  }

  }
}
// CHECK59: [[@LINE-31]]:7 -> [[@LINE-31]]:12, [[@LINE-30]]:1 -> [[@LINE-30]]:9, [[@LINE-30]]:31 -> [[@LINE-30]]:36
+(some_type_t)withSomething:(BOOL)bar
z_Z_42:(BOOL)struct {
  [self usingThing: (globalArray[i]) world: @"string literal" name: [super test: globalArray[i] part: ^ () {
   [super test: 12 z_Z_42: "string"];

 } perform: ']' perform: "string"] object: "]" name: 12];

  /*comment*/[self world: "string" bar: ^  {
   [self.undef_property bar: @"string literal" name: ^  {


 } z_Z_42: "string" + [_undef_ivar name: [self part: 12 struct: []  {
  }
] withSomething: @"string literal" class: ']' < "string" == "]" < "string" == "]"] onEntity: globalArray[i] object: "]"];

 } == ']' test: @"string literal" world: "]" struct: "string"];
}
// CHECK60: [[@LINE-17]]:15 -> [[@LINE-17]]:28, [[@LINE-16]]:1 -> [[@LINE-16]]:7
@end
// RUN: clang-refactor-test rename-indexed-file -name=piece:perform:onEntity:bar:usingThing:class -new-name=method:bar:piece:onEntity:foo:piece -indexed-file=%s -indexed-at=602:8 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK57 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:usingThing:perform -new-name=name:usingThing:foo -indexed-file=%s -indexed-at=623:7 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK58 %s
// RUN: clang-refactor-test rename-indexed-file -name=piece:onEntity:piece -new-name=class:part:world -indexed-file=%s -indexed-at=636:7 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK59 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:z_Z_42 -new-name=object:object -indexed-file=%s -indexed-at=668:15 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK60 %s

@interface test (z_Z_42)
-(void)struct:(some_type_t)onEntity perform:(void)struct
usingThing:(BOOL)foo onEntity:(some_type_t)onEntity /*comment*/ a_200:(Object *)piece ;
// CHECK61: [[@LINE-2]]:8 -> [[@LINE-2]]:14, [[@LINE-2]]:37 -> [[@LINE-2]]:44, [[@LINE-1]]:1 -> [[@LINE-1]]:11, [[@LINE-1]]:22 -> [[@LINE-1]]:30, [[@LINE-1]]:65 -> [[@LINE-1]]:70
@end
// RUN: clang-refactor-test rename-indexed-file -name=struct:perform:usingThing:onEntity:a_200 -new-name=method:world:foo:foo:perform -indexed-file=%s -indexed-at=693:8 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK61 %s

@implementation part
-(some_type_t)bar:(some_type_t (^)(some_type_t, Object *))class piece:(some_type_t)z_Z_42
test:(BOOL)z_Z_42 /*comment*/ class:(void)class __attribute__((test())){
  [globalObject send: [self onEntity:  == ']' z_Z_42: []  {    int bar = [_undef_ivar a_200: [super a_200: @"string literal"
//comment
 withSomething: @"string literal" + ^  {
  }] foo: "]"] other: 42];
  } test: ^ () {    if ([] () {


 }) {
      int bar = [self part: ^ () {
  }
 class: [] () {
  } object: @"string literal"];

  }
  }];

  [globalObject message] = [_undef_ivar part: (@{ @1, @3 }) struct: "]" z_Z_42: [self perform: @"string literal" usingThing: ']' a_200: ^  {    // comment
  } struct: ^ () {
   if ("string") {
      int test = "string";

  }

 }]];
}
// CHECK62: [[@LINE-27]]:15 -> [[@LINE-27]]:18, [[@LINE-27]]:65 -> [[@LINE-27]]:70, [[@LINE-26]]:1 -> [[@LINE-26]]:5, [[@LINE-26]]:31 -> [[@LINE-26]]:36
@end
// RUN: clang-refactor-test rename-indexed-file -name=bar:piece:test:class -new-name=perform:bar:struct:world -indexed-file=%s -indexed-at=700:15 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK62 %s

@interface perform
+(void)piece:(BOOL)a_200 onEntity:(const Object & (^)())object bar:(int (*)())method //comment
method:(some_type_t)onEntity
;
// CHECK63: [[@LINE-3]]:8 -> [[@LINE-3]]:13, [[@LINE-3]]:26 -> [[@LINE-3]]:34, [[@LINE-3]]:64 -> [[@LINE-3]]:67, [[@LINE-2]]:1 -> [[@LINE-2]]:7
-(void)withSomething:(const Object &)piece //comment
piece:(void)perform ;
// CHECK64: [[@LINE-2]]:8 -> [[@LINE-2]]:21, [[@LINE-1]]:1 -> [[@LINE-1]]:6
-(some_type_t)z_Z_42:(int)method piece:(^ { })struct struct:(BOOL)world /*comment*/ a_200:(const Object &)piece
;
// CHECK65: [[@LINE-2]]:15 -> [[@LINE-2]]:21, [[@LINE-2]]:34 -> [[@LINE-2]]:39, [[@LINE-2]]:54 -> [[@LINE-2]]:60, [[@LINE-2]]:85 -> [[@LINE-2]]:90
+(Object *)foo:(const Object &)part bar:(some_type_t)a_200 ;
// CHECK66: [[@LINE-1]]:12 -> [[@LINE-1]]:15, [[@LINE-1]]:37 -> [[@LINE-1]]:40
@end
// RUN: clang-refactor-test rename-indexed-file -name=piece:onEntity:bar:method -new-name=onEntity:withSomething:class:piece -indexed-file=%s -indexed-at=732:8 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK63 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:piece -new-name=perform:foo -indexed-file=%s -indexed-at=736:8 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK64 %s
// RUN: clang-refactor-test rename-indexed-file -name=z_Z_42:piece:struct:a_200 -new-name=test:withSomething:piece:class -indexed-file=%s -indexed-at=739:15 -indexed-symbol-kind=objc-im %s | FileCheck --check-prefix=CHECK65 %s
// RUN: clang-refactor-test rename-indexed-file -name=foo:bar -new-name=usingThing:a_200 -indexed-file=%s -indexed-at=742:12 -indexed-symbol-kind=objc-cm %s | FileCheck --check-prefix=CHECK66 %s
