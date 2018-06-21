+(BOOL) onEntity {
  call() = [_undef_ivar name: @"string literal" method: @"string literal"];
  // CHECK1: [[@LINE-1]]:25 -> [[@LINE-1]]:29, [[@LINE-1]]:49 -> [[@LINE-1]]:55
}
// RUN: clang-refactor-test rename-indexed-file -name=name:method -new-name=object:world -indexed-file=%s -indexed-at=2:25 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK1 %s

-(const Object &) a_200 {
  some_type_t world = @"string literal";
  [self withSomething: [] () {
   some_type_t foo = [super struct: @"string literal"
 world: [] () {
  } name: []  {


 } a_200: 12 a_200: globalArray[i]];
  // CHECK2: [[@LINE-6]]:29 -> [[@LINE-6]]:35, [[@LINE-5]]:2 -> [[@LINE-5]]:7, [[@LINE-4]]:5 -> [[@LINE-4]]:9, [[@LINE-1]]:4 -> [[@LINE-1]]:9, [[@LINE-1]]:14 -> [[@LINE-1]]:19

 } onEntity: ']' perform: "]"];
  // CHECK3: [[@LINE-10]]:9 -> [[@LINE-10]]:22, [[@LINE-1]]:4 -> [[@LINE-1]]:12, [[@LINE-1]]:18 -> [[@LINE-1]]:25
}
// RUN: clang-refactor-test rename-indexed-file -name=struct:world:name:a_200:a_200 -new-name=withSomething:foo:part:struct:method -indexed-file=%s -indexed-at=10:29 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK2 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:onEntity:perform -new-name=method:foo:name -indexed-file=%s -indexed-at=9:9 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK3 %s

-(int) struct {
  // comment
}

-(int) bar {
  return ']';
}

-(int) part {
  call() = [self test: 12 method: globalArray[i] test: globalArray[i] world: ']'];
  // CHECK4: [[@LINE-1]]:18 -> [[@LINE-1]]:22, [[@LINE-1]]:27 -> [[@LINE-1]]:33, [[@LINE-1]]:50 -> [[@LINE-1]]:54, [[@LINE-1]]:71 -> [[@LINE-1]]:76
  [_undef_ivar withSomething: @{ @1, @3 } onEntity: [] () {    ([self foo: "]" piece:  foo: (']')]);
  // CHECK5: [[@LINE-1]]:71 -> [[@LINE-1]]:74, [[@LINE-1]]:80 -> [[@LINE-1]]:85, [[@LINE-1]]:88 -> [[@LINE-1]]:91
  }];
  // CHECK6: [[@LINE-3]]:16 -> [[@LINE-3]]:29, [[@LINE-3]]:43 -> [[@LINE-3]]:51
  int bar = [self name: ']' z_Z_42: [] () {    [globalObject send: [self perform: globalArray[i] * "string" a_200: 12 class: 12 perform: ']' object: [] () {
  }] other: 42];
  // CHECK7: [[@LINE-2]]:74 -> [[@LINE-2]]:81, [[@LINE-2]]:109 -> [[@LINE-2]]:114, [[@LINE-2]]:119 -> [[@LINE-2]]:124, [[@LINE-2]]:129 -> [[@LINE-2]]:136, [[@LINE-2]]:142 -> [[@LINE-2]]:148
  } bar: ^  {
   [globalObject message] = ([self onEntity: globalArray[i] class: 12 foo: "string"]);
  // CHECK8: [[@LINE-1]]:36 -> [[@LINE-1]]:44, [[@LINE-1]]:61 -> [[@LINE-1]]:66, [[@LINE-1]]:71 -> [[@LINE-1]]:74

 } bar: ^ () {
   some_type_t foo = [self perform: 12 bar: (^ () {
  }) bar: ^  {
  } perform: @"string literal" test: "]" + 12];
  // CHECK9: [[@LINE-3]]:28 -> [[@LINE-3]]:35, [[@LINE-3]]:40 -> [[@LINE-3]]:43, [[@LINE-2]]:6 -> [[@LINE-2]]:9, [[@LINE-1]]:5 -> [[@LINE-1]]:12, [[@LINE-1]]:32 -> [[@LINE-1]]:36

 }];
  // CHECK10: [[@LINE-14]]:19 -> [[@LINE-14]]:23, [[@LINE-14]]:29 -> [[@LINE-14]]:35, [[@LINE-11]]:5 -> [[@LINE-11]]:8, [[@LINE-7]]:4 -> [[@LINE-7]]:7
  [self piece: @"string literal" method: globalArray[i] method: "string" class: globalArray[i]];
  // CHECK11: [[@LINE-1]]:9 -> [[@LINE-1]]:14, [[@LINE-1]]:34 -> [[@LINE-1]]:40, [[@LINE-1]]:57 -> [[@LINE-1]]:63, [[@LINE-1]]:74 -> [[@LINE-1]]:79
}
// RUN: clang-refactor-test rename-indexed-file -name=test:method:test:world -new-name=a_200:struct:perform:piece -indexed-file=%s -indexed-at=33:18 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK4 %s
// RUN: clang-refactor-test rename-indexed-file -name=foo:piece:foo -new-name=world:onEntity:name -indexed-file=%s -indexed-at=35:71 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK5 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:onEntity -new-name=class:object -indexed-file=%s -indexed-at=35:16 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK6 %s
// RUN: clang-refactor-test rename-indexed-file -name=perform:a_200:class:perform:object -new-name=method:perform:class:foo:name -indexed-file=%s -indexed-at=39:74 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK7 %s
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:class:foo -new-name=piece:onEntity:bar -indexed-file=%s -indexed-at=43:36 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK8 %s
// RUN: clang-refactor-test rename-indexed-file -name=perform:bar:bar:perform:test -new-name=foo:world:class:struct:z_Z_42 -indexed-file=%s -indexed-at=47:28 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK9 %s
// RUN: clang-refactor-test rename-indexed-file -name=name:z_Z_42:bar:bar -new-name=world:piece:perform:test -indexed-file=%s -indexed-at=39:19 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK10 %s
// RUN: clang-refactor-test rename-indexed-file -name=piece:method:method:class -new-name=a_200:withSomething:onEntity:onEntity -indexed-file=%s -indexed-at=54:9 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK11 %s

+(int) struct {
  call() = [self name: ^ () {    const Object & piece = 12;
  } perform: @{ @1, @3 } a_200: ^  {
   some_type_t foo = [self name: ']' name: ^ () {
  } method: ']'];
  // CHECK12: [[@LINE-2]]:28 -> [[@LINE-2]]:32, [[@LINE-2]]:38 -> [[@LINE-2]]:42, [[@LINE-1]]:5 -> [[@LINE-1]]:11

 }];
  // CHECK13: [[@LINE-7]]:18 -> [[@LINE-7]]:22, [[@LINE-6]]:5 -> [[@LINE-6]]:12, [[@LINE-6]]:26 -> [[@LINE-6]]:31
}
// RUN: clang-refactor-test rename-indexed-file -name=name:name:method -new-name=part:bar:usingThing -indexed-file=%s -indexed-at=69:28 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK12 %s
// RUN: clang-refactor-test rename-indexed-file -name=name:perform:a_200 -new-name=a_200:piece:class -indexed-file=%s -indexed-at=67:18 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK13 %s

+(BOOL) world {
  int bar = ([_undef_ivar world: "string" struct: @"string literal" world: globalArray[i]
 perform: ^  {    // comment
  } < "string" name: [] () {
   int bar = [self a_200: "string" * 12 method: globalArray[i] usingThing: @"string literal" part: ']'];
  // CHECK14: [[@LINE-1]]:20 -> [[@LINE-1]]:25, [[@LINE-1]]:41 -> [[@LINE-1]]:47, [[@LINE-1]]:64 -> [[@LINE-1]]:74, [[@LINE-1]]:94 -> [[@LINE-1]]:98

 }]);
  // CHECK15: [[@LINE-7]]:27 -> [[@LINE-7]]:32, [[@LINE-7]]:43 -> [[@LINE-7]]:49, [[@LINE-7]]:69 -> [[@LINE-7]]:74, [[@LINE-6]]:2 -> [[@LINE-6]]:9, [[@LINE-5]]:16 -> [[@LINE-5]]:20
  [self perform: [self object: globalArray[i] onEntity: /*]*/ [] () {
   ; ;[self.undef_property test: [] () {


 } onEntity: @"string literal"
];
  // CHECK16: [[@LINE-5]]:28 -> [[@LINE-5]]:32, [[@LINE-2]]:4 -> [[@LINE-2]]:12

 } method: "]"] world: "]" withSomething: @"string literal"];
  // CHECK17: [[@LINE-9]]:24 -> [[@LINE-9]]:30, [[@LINE-9]]:47 -> [[@LINE-9]]:55, [[@LINE-1]]:4 -> [[@LINE-1]]:10
  // CHECK18: [[@LINE-10]]:9 -> [[@LINE-10]]:16, [[@LINE-2]]:17 -> [[@LINE-2]]:22, [[@LINE-2]]:28 -> [[@LINE-2]]:41
  [self usingThing: ^  {
   ;

 } world: "string" test: "]"];
  // CHECK19: [[@LINE-4]]:9 -> [[@LINE-4]]:19, [[@LINE-1]]:4 -> [[@LINE-1]]:9, [[@LINE-1]]:20 -> [[@LINE-1]]:24
  call() = [self bar: []  {    globalArray[12] = [self onEntity: ^ () { ] } foo: "string" piece: @{ @1, @3 } bar: ']'];
  // CHECK20: [[@LINE-1]]:56 -> [[@LINE-1]]:64, [[@LINE-1]]:77 -> [[@LINE-1]]:80, [[@LINE-1]]:91 -> [[@LINE-1]]:96, [[@LINE-1]]:110 -> [[@LINE-1]]:113
  }
//comment
 struct: []  {    return ^ () {
  };
  } piece: "string" onEntity: ^ () { ] }];
  // CHECK21: [[@LINE-7]]:18 -> [[@LINE-7]]:21, [[@LINE-3]]:2 -> [[@LINE-3]]:8, [[@LINE-1]]:5 -> [[@LINE-1]]:10, [[@LINE-1]]:21 -> [[@LINE-1]]:29
  some_type_t foo = [_undef_ivar world: 12
 bar: [] () {
   [globalObject send: [super class: 12 world: [self perform: 12 perform: @{ @1, @3 }
 object: "string"] class: "string" z_Z_42: @{ @1, @3 }] other: 42];
  // CHECK22: [[@LINE-2]]:54 -> [[@LINE-2]]:61, [[@LINE-2]]:66 -> [[@LINE-2]]:73, [[@LINE-1]]:2 -> [[@LINE-1]]:8
  // CHECK23: [[@LINE-3]]:31 -> [[@LINE-3]]:36, [[@LINE-3]]:41 -> [[@LINE-3]]:46, [[@LINE-2]]:20 -> [[@LINE-2]]:25, [[@LINE-2]]:36 -> [[@LINE-2]]:42

 }];
  // CHECK24: [[@LINE-8]]:34 -> [[@LINE-8]]:39, [[@LINE-7]]:2 -> [[@LINE-7]]:5
}
// RUN: clang-refactor-test rename-indexed-file -name=a_200:method:usingThing:part -new-name=foo:bar:bar:class -indexed-file=%s -indexed-at=83:20 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK14 %s
// RUN: clang-refactor-test rename-indexed-file -name=world:struct:world:perform:name -new-name=perform:bar:object:foo:object -indexed-file=%s -indexed-at=80:27 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK15 %s
// RUN: clang-refactor-test rename-indexed-file -name=test:onEntity -new-name=name:class -indexed-file=%s -indexed-at=89:28 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK16 %s
// RUN: clang-refactor-test rename-indexed-file -name=object:onEntity:method -new-name=usingThing:a_200:onEntity -indexed-file=%s -indexed-at=88:24 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK17 %s
// RUN: clang-refactor-test rename-indexed-file -name=perform:world:withSomething -new-name=onEntity:method:part -indexed-file=%s -indexed-at=88:9 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK18 %s
// RUN: clang-refactor-test rename-indexed-file -name=usingThing:world:test -new-name=name:object:onEntity -indexed-file=%s -indexed-at=99:9 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK19 %s
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:foo:piece:bar -new-name=test:foo:test:name -indexed-file=%s -indexed-at=104:56 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK20 %s
// RUN: clang-refactor-test rename-indexed-file -name=bar:struct:piece:onEntity -new-name=usingThing:method:part:piece -indexed-file=%s -indexed-at=104:18 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK21 %s
// RUN: clang-refactor-test rename-indexed-file -name=perform:perform:object -new-name=usingThing:withSomething:withSomething -indexed-file=%s -indexed-at=114:54 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK22 %s
// RUN: clang-refactor-test rename-indexed-file -name=class:world:class:z_Z_42 -new-name=bar:piece:class:z_Z_42 -indexed-file=%s -indexed-at=114:31 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK23 %s
// RUN: clang-refactor-test rename-indexed-file -name=world:bar -new-name=foo:bar -indexed-file=%s -indexed-at=112:34 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK24 %s

-(void) test {
  call() = [self name: globalArray[i] onEntity: "]" bar: ^  {    return [self withSomething: [] () {


 } name: ']' part: 12];
  // CHECK25: [[@LINE-4]]:79 -> [[@LINE-4]]:92, [[@LINE-1]]:4 -> [[@LINE-1]]:8, [[@LINE-1]]:14 -> [[@LINE-1]]:18
  } usingThing: "]" object: ];
  // CHECK26: [[@LINE-6]]:18 -> [[@LINE-6]]:22, [[@LINE-6]]:39 -> [[@LINE-6]]:47, [[@LINE-6]]:53 -> [[@LINE-6]]:56, [[@LINE-1]]:5 -> [[@LINE-1]]:15, [[@LINE-1]]:21 -> [[@LINE-1]]:27
  [globalObject message] = [self a_200: globalArray[i] struct: @{ @1, @3 }
];
  // CHECK27: [[@LINE-2]]:34 -> [[@LINE-2]]:39, [[@LINE-2]]:56 -> [[@LINE-2]]:62
}
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:name:part -new-name=z_Z_42:object:test -indexed-file=%s -indexed-at=135:79 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK25 %s
// RUN: clang-refactor-test rename-indexed-file -name=name:onEntity:bar:usingThing:object -new-name=foo:method:perform:onEntity:perform -indexed-file=%s -indexed-at=135:18 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK26 %s
// RUN: clang-refactor-test rename-indexed-file -name=a_200:struct -new-name=bar:perform -indexed-file=%s -indexed-at=142:34 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK27 %s

-(int) class {
  return /*]*/ [] () {
   call() = [self struct: (@"string literal") method: ^ () {
  } foo: @"string literal" onEntity: []  {


 } < globalArray[i] method: ']'];
  // CHECK28: [[@LINE-5]]:19 -> [[@LINE-5]]:25, [[@LINE-5]]:47 -> [[@LINE-5]]:53, [[@LINE-4]]:5 -> [[@LINE-4]]:8, [[@LINE-4]]:28 -> [[@LINE-4]]:36, [[@LINE-1]]:21 -> [[@LINE-1]]:27

 };
  [self name: [] () {    if ([]  {


 }) {
      BOOL class = @{ @1, @3 };

  }
  } perform: 12 onEntity: @"string literal"];
  // CHECK29: [[@LINE-8]]:9 -> [[@LINE-8]]:13, [[@LINE-1]]:5 -> [[@LINE-1]]:12, [[@LINE-1]]:17 -> [[@LINE-1]]:25
  return globalArray[i] * [self withSomething: [self part: 12 name: ^  {    [globalObject send: [super withSomething: [self test: [] () {


 } world: [_undef_ivar piece: 12 class: @"string literal" test: "string" bar: /*]*/ globalArray[i]]
] withSomething: 12 name: ']' test: "]" usingThing: @"string literal"] other: 42];
  // CHECK30: [[@LINE-2]]:24 -> [[@LINE-2]]:29, [[@LINE-2]]:34 -> [[@LINE-2]]:39, [[@LINE-2]]:59 -> [[@LINE-2]]:63, [[@LINE-2]]:74 -> [[@LINE-2]]:77
  // CHECK31: [[@LINE-6]]:125 -> [[@LINE-6]]:129, [[@LINE-3]]:4 -> [[@LINE-3]]:9
  // CHECK32: [[@LINE-7]]:104 -> [[@LINE-7]]:117, [[@LINE-3]]:3 -> [[@LINE-3]]:16, [[@LINE-3]]:21 -> [[@LINE-3]]:25, [[@LINE-3]]:31 -> [[@LINE-3]]:35, [[@LINE-3]]:41 -> [[@LINE-3]]:51
  } part: @{ @1, @3 }] withSomething: ']' foo: globalArray[i]];
  // CHECK33: [[@LINE-9]]:54 -> [[@LINE-9]]:58, [[@LINE-9]]:63 -> [[@LINE-9]]:67, [[@LINE-1]]:5 -> [[@LINE-1]]:9
  // CHECK34: [[@LINE-10]]:33 -> [[@LINE-10]]:46, [[@LINE-2]]:24 -> [[@LINE-2]]:37, [[@LINE-2]]:43 -> [[@LINE-2]]:46
}
// RUN: clang-refactor-test rename-indexed-file -name=struct:method:foo:onEntity:method -new-name=object:piece:struct:foo:name -indexed-file=%s -indexed-at=152:19 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK28 %s
// RUN: clang-refactor-test rename-indexed-file -name=name:perform:onEntity -new-name=z_Z_42:bar:z_Z_42 -indexed-file=%s -indexed-at=160:9 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK29 %s
// RUN: clang-refactor-test rename-indexed-file -name=piece:class:test:bar -new-name=world:bar:object:perform -indexed-file=%s -indexed-at=172:24 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK30 %s
// RUN: clang-refactor-test rename-indexed-file -name=test:world -new-name=a_200:bar -indexed-file=%s -indexed-at=169:125 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK31 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:withSomething:name:test:usingThing -new-name=bar:class:class:perform:perform -indexed-file=%s -indexed-at=169:104 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK32 %s
// RUN: clang-refactor-test rename-indexed-file -name=part:name:part -new-name=class:perform:name -indexed-file=%s -indexed-at=169:54 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK33 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:withSomething:foo -new-name=test:object:withSomething -indexed-file=%s -indexed-at=169:33 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK34 %s

+(Object *) usingThing {
  return "string";
}

+(void) class {
  if (@{ @1, @3 }) {
      globalArray[12] = [self foo: []  {    [globalObject message] = [self object: @{ @1, @3 } usingThing: globalArray[i] perform: "]"];
  // CHECK35: [[@LINE-1]]:76 -> [[@LINE-1]]:82, [[@LINE-1]]:96 -> [[@LINE-1]]:106, [[@LINE-1]]:123 -> [[@LINE-1]]:130
  } class:  usingThing: "]" perform: [self.undef_property name: ^ () { ] } piece: 12 name: ^ () {
   globalArray[12] = [_undef_ivar foo: ']' foo: []  {
  }
 bar: ^  {
  } + []  {


 }
];
  // CHECK36: [[@LINE-8]]:35 -> [[@LINE-8]]:38, [[@LINE-8]]:44 -> [[@LINE-8]]:47, [[@LINE-6]]:2 -> [[@LINE-6]]:5

 }]];
  // CHECK37: [[@LINE-12]]:59 -> [[@LINE-12]]:63, [[@LINE-12]]:76 -> [[@LINE-12]]:81, [[@LINE-12]]:86 -> [[@LINE-12]]:90
  // CHECK38: [[@LINE-15]]:31 -> [[@LINE-15]]:34, [[@LINE-13]]:5 -> [[@LINE-13]]:10, [[@LINE-13]]:13 -> [[@LINE-13]]:23, [[@LINE-13]]:29 -> [[@LINE-13]]:36

  }
}
// RUN: clang-refactor-test rename-indexed-file -name=object:usingThing:perform -new-name=test:object:z_Z_42 -indexed-file=%s -indexed-at=195:76 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK35 %s
// RUN: clang-refactor-test rename-indexed-file -name=foo:foo:bar -new-name=method:part:class -indexed-file=%s -indexed-at=198:35 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK36 %s
// RUN: clang-refactor-test rename-indexed-file -name=name:piece:name -new-name=perform:onEntity:struct -indexed-file=%s -indexed-at=197:59 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK37 %s
// RUN: clang-refactor-test rename-indexed-file -name=foo:class:usingThing:perform -new-name=name:z_Z_42:method:test -indexed-file=%s -indexed-at=195:31 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK38 %s

-(some_type_t) struct {
  globalArray[12] = [super a_200: globalArray[i]
 name: globalArray[i] world: []  {
   [globalObject send: [self part: ']' z_Z_42: ^  {
  } + "]" struct: "string"
 bar: [] () {


 }] other: 42];
  // CHECK39: [[@LINE-6]]:30 -> [[@LINE-6]]:34, [[@LINE-6]]:40 -> [[@LINE-6]]:46, [[@LINE-5]]:11 -> [[@LINE-5]]:17, [[@LINE-4]]:2 -> [[@LINE-4]]:5

 } onEntity: ']' == globalArray[i]];
  // CHECK40: [[@LINE-11]]:28 -> [[@LINE-11]]:33, [[@LINE-10]]:2 -> [[@LINE-10]]:6, [[@LINE-10]]:23 -> [[@LINE-10]]:28, [[@LINE-1]]:4 -> [[@LINE-1]]:12
  [self a_200: [] () {    BOOL class = 12;
  } part: ']' test: ^ () {    int z_Z_42 = [self struct: "]" perform: "]" method: globalArray[i]];
  // CHECK41: [[@LINE-1]]:50 -> [[@LINE-1]]:56, [[@LINE-1]]:62 -> [[@LINE-1]]:69, [[@LINE-1]]:75 -> [[@LINE-1]]:81
  }];
  // CHECK42: [[@LINE-4]]:9 -> [[@LINE-4]]:14, [[@LINE-3]]:5 -> [[@LINE-3]]:9, [[@LINE-3]]:15 -> [[@LINE-3]]:19
}
// RUN: clang-refactor-test rename-indexed-file -name=part:z_Z_42:struct:bar -new-name=world:a_200:a_200:test -indexed-file=%s -indexed-at=222:30 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK39 %s
// RUN: clang-refactor-test rename-indexed-file -name=a_200:name:world:onEntity -new-name=object:object:perform:z_Z_42 -indexed-file=%s -indexed-at=220:28 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK40 %s
// RUN: clang-refactor-test rename-indexed-file -name=struct:perform:method -new-name=perform:test:bar -indexed-file=%s -indexed-at=233:50 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK41 %s
// RUN: clang-refactor-test rename-indexed-file -name=a_200:part:test -new-name=class:method:test -indexed-file=%s -indexed-at=232:9 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK42 %s

+(some_type_t) piece {
  call() = [self class: "string" part: 12];
  // CHECK43: [[@LINE-1]]:18 -> [[@LINE-1]]:23, [[@LINE-1]]:34 -> [[@LINE-1]]:38
  [self struct: globalArray[i] part: "]" part: [self method: @"string literal" == globalArray[i] + [self onEntity: ^  {    [_undef_ivar piece: @{ @1, @3 } world: globalArray[i]];
  // CHECK44: [[@LINE-1]]:137 -> [[@LINE-1]]:142, [[@LINE-1]]:156 -> [[@LINE-1]]:161
  } object: (^ () {    [self class: 12 object: []  {


 } == ^ () {


 }];
  // CHECK45: [[@LINE-7]]:30 -> [[@LINE-7]]:35, [[@LINE-7]]:40 -> [[@LINE-7]]:46
  }) piece: ']' world: 12 part: (12)] struct: /*]*/ globalArray[i] foo: [self object: ^  {    // comment
  // CHECK46: [[@LINE-11]]:106 -> [[@LINE-11]]:114, [[@LINE-9]]:5 -> [[@LINE-9]]:11, [[@LINE-1]]:6 -> [[@LINE-1]]:11, [[@LINE-1]]:17 -> [[@LINE-1]]:22, [[@LINE-1]]:27 -> [[@LINE-1]]:31
  } world: (^  {
   int bar = [_undef_ivar test: globalArray[i] + "string" == ']' class: 12];
  // CHECK47: [[@LINE-1]]:27 -> [[@LINE-1]]:31, [[@LINE-1]]:66 -> [[@LINE-1]]:71

 }) part: ']'
] perform: "]"] perform: @{ @1, @3 }];
  // CHECK48: [[@LINE-8]]:79 -> [[@LINE-8]]:85, [[@LINE-6]]:5 -> [[@LINE-6]]:10, [[@LINE-2]]:5 -> [[@LINE-2]]:9
  // CHECK49: [[@LINE-19]]:54 -> [[@LINE-19]]:60, [[@LINE-9]]:39 -> [[@LINE-9]]:45, [[@LINE-9]]:68 -> [[@LINE-9]]:71, [[@LINE-2]]:3 -> [[@LINE-2]]:10
  // CHECK50: [[@LINE-20]]:9 -> [[@LINE-20]]:15, [[@LINE-20]]:32 -> [[@LINE-20]]:36, [[@LINE-20]]:42 -> [[@LINE-20]]:46, [[@LINE-3]]:17 -> [[@LINE-3]]:24
  some_type_t foo = [self.undef_property class: "]" part: []  {
   [self bar: [] () {
  } z_Z_42: "]" name: globalArray[i]];
  // CHECK51: [[@LINE-2]]:10 -> [[@LINE-2]]:13, [[@LINE-1]]:5 -> [[@LINE-1]]:11, [[@LINE-1]]:17 -> [[@LINE-1]]:21

 }
 onEntity: ^  {    return [super withSomething: "string" usingThing: @"string literal" test: @"string literal" withSomething: ']' world: [] () {


 }
];
  // CHECK52: [[@LINE-5]]:34 -> [[@LINE-5]]:47, [[@LINE-5]]:58 -> [[@LINE-5]]:68, [[@LINE-5]]:88 -> [[@LINE-5]]:92, [[@LINE-5]]:112 -> [[@LINE-5]]:125, [[@LINE-5]]:131 -> [[@LINE-5]]:136
  } class: 12 struct: @"string literal" == globalArray[i]];
  // CHECK53: [[@LINE-13]]:42 -> [[@LINE-13]]:47, [[@LINE-13]]:53 -> [[@LINE-13]]:57, [[@LINE-7]]:2 -> [[@LINE-7]]:10, [[@LINE-1]]:5 -> [[@LINE-1]]:10, [[@LINE-1]]:15 -> [[@LINE-1]]:21
}
// RUN: clang-refactor-test rename-indexed-file -name=class:part -new-name=a_200:class -indexed-file=%s -indexed-at=244:18 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK43 %s
// RUN: clang-refactor-test rename-indexed-file -name=piece:world -new-name=part:name -indexed-file=%s -indexed-at=246:137 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK44 %s
// RUN: clang-refactor-test rename-indexed-file -name=class:object -new-name=onEntity:z_Z_42 -indexed-file=%s -indexed-at=248:30 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK45 %s
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:object:piece:world:part -new-name=onEntity:piece:a_200:class:struct -indexed-file=%s -indexed-at=246:106 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK46 %s
// RUN: clang-refactor-test rename-indexed-file -name=test:class -new-name=a_200:test -indexed-file=%s -indexed-at=259:27 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK47 %s
// RUN: clang-refactor-test rename-indexed-file -name=object:world:part -new-name=class:z_Z_42:onEntity -indexed-file=%s -indexed-at=256:79 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK48 %s
// RUN: clang-refactor-test rename-indexed-file -name=method:struct:foo:perform -new-name=onEntity:bar:part:test -indexed-file=%s -indexed-at=246:54 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK49 %s
// RUN: clang-refactor-test rename-indexed-file -name=struct:part:part:perform -new-name=piece:object:world:usingThing -indexed-file=%s -indexed-at=246:9 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK50 %s
// RUN: clang-refactor-test rename-indexed-file -name=bar:z_Z_42:name -new-name=bar:test:struct -indexed-file=%s -indexed-at=268:10 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK51 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:usingThing:test:withSomething:world -new-name=foo:test:usingThing:piece:piece -indexed-file=%s -indexed-at=273:34 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK52 %s
// RUN: clang-refactor-test rename-indexed-file -name=class:part:onEntity:class:struct -new-name=z_Z_42:onEntity:onEntity:a_200:class -indexed-file=%s -indexed-at=267:42 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK53 %s

+(void) z_Z_42 {
  int a_200 = @{ @1, @3 };
  // comment
  [globalObject send: ([self onEntity: [] () {    [super onEntity: ^  {
  } class: @"string literal" perform: @"string literal"] other: 42];
  // CHECK54: [[@LINE-2]]:58 -> [[@LINE-2]]:66, [[@LINE-1]]:5 -> [[@LINE-1]]:10, [[@LINE-1]]:30 -> [[@LINE-1]]:37
  } class: globalArray[i] a_200: ^ () { ] }
]);
  // CHECK55: [[@LINE-5]]:30 -> [[@LINE-5]]:38, [[@LINE-2]]:5 -> [[@LINE-2]]:10, [[@LINE-2]]:27 -> [[@LINE-2]]:32
  globalArray[12] = [super method: @"string literal" test: ^ () {    [self struct: "]" method: globalArray[i]];
  // CHECK56: [[@LINE-1]]:76 -> [[@LINE-1]]:82, [[@LINE-1]]:88 -> [[@LINE-1]]:94
  } name: globalArray[i]];
  // CHECK57: [[@LINE-3]]:28 -> [[@LINE-3]]:34, [[@LINE-3]]:54 -> [[@LINE-3]]:58, [[@LINE-1]]:5 -> [[@LINE-1]]:9
  BOOL struct = 12;
}
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:class:perform -new-name=bar:z_Z_42:onEntity -indexed-file=%s -indexed-at=297:58 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK54 %s
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:class:a_200 -new-name=withSomething:piece:foo -indexed-file=%s -indexed-at=297:30 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK55 %s
// RUN: clang-refactor-test rename-indexed-file -name=struct:method -new-name=onEntity:struct -indexed-file=%s -indexed-at=303:76 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK56 %s
// RUN: clang-refactor-test rename-indexed-file -name=method:test:name -new-name=name:z_Z_42:object -indexed-file=%s -indexed-at=303:28 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK57 %s

+(int) onEntity {
  call() = [self piece: []  {    return "string";
  } foo: ']' bar: globalArray[i]];
  // CHECK58: [[@LINE-2]]:18 -> [[@LINE-2]]:23, [[@LINE-1]]:5 -> [[@LINE-1]]:8, [[@LINE-1]]:14 -> [[@LINE-1]]:17
  [globalObject send: [self usingThing: []  {
   [_undef_ivar test: globalArray[i] part: 12 world: "string" onEntity: 12] other: 42];
  // CHECK59: [[@LINE-1]]:17 -> [[@LINE-1]]:21, [[@LINE-1]]:38 -> [[@LINE-1]]:42, [[@LINE-1]]:47 -> [[@LINE-1]]:52, [[@LINE-1]]:63 -> [[@LINE-1]]:71

 } a_200: @"string literal" a_200: "string" object: ("string")
//comment
];
  // CHECK60: [[@LINE-7]]:29 -> [[@LINE-7]]:39, [[@LINE-3]]:4 -> [[@LINE-3]]:9, [[@LINE-3]]:29 -> [[@LINE-3]]:34, [[@LINE-3]]:45 -> [[@LINE-3]]:51
  [globalObject send: [super object: ']' name: ^  {
   [globalObject send: [self.undef_property bar:  == [self name: "string" bar: ']']
 part: @"string literal" z_Z_42: ']' part: ']' test: "string"] other: 42] other: 42];
  // CHECK61: [[@LINE-2]]:60 -> [[@LINE-2]]:64, [[@LINE-2]]:75 -> [[@LINE-2]]:78
  // CHECK62: [[@LINE-3]]:45 -> [[@LINE-3]]:48, [[@LINE-2]]:2 -> [[@LINE-2]]:6, [[@LINE-2]]:26 -> [[@LINE-2]]:32, [[@LINE-2]]:38 -> [[@LINE-2]]:42, [[@LINE-2]]:48 -> [[@LINE-2]]:52

 }
 object: globalArray[i] foo: (']')];
  // CHECK63: [[@LINE-8]]:30 -> [[@LINE-8]]:36, [[@LINE-8]]:42 -> [[@LINE-8]]:46, [[@LINE-1]]:2 -> [[@LINE-1]]:8, [[@LINE-1]]:25 -> [[@LINE-1]]:28
  [self.undef_property world: globalArray[i] onEntity: ']' object: @"string literal" == 12 struct: []  {
   int bar = [self onEntity: "]" piece: [] () {
  }];
  // CHECK64: [[@LINE-2]]:20 -> [[@LINE-2]]:28, [[@LINE-2]]:34 -> [[@LINE-2]]:39

 } struct: @"string literal"];
  // CHECK65: [[@LINE-6]]:24 -> [[@LINE-6]]:29, [[@LINE-6]]:46 -> [[@LINE-6]]:54, [[@LINE-6]]:60 -> [[@LINE-6]]:66, [[@LINE-6]]:92 -> [[@LINE-6]]:98, [[@LINE-1]]:4 -> [[@LINE-1]]:10
}
// RUN: clang-refactor-test rename-indexed-file -name=piece:foo:bar -new-name=class:onEntity:method -indexed-file=%s -indexed-at=315:18 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK58 %s
// RUN: clang-refactor-test rename-indexed-file -name=test:part:world:onEntity -new-name=z_Z_42:bar:piece:perform -indexed-file=%s -indexed-at=319:17 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK59 %s
// RUN: clang-refactor-test rename-indexed-file -name=usingThing:a_200:a_200:object -new-name=a_200:object:method:perform -indexed-file=%s -indexed-at=318:29 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK60 %s
// RUN: clang-refactor-test rename-indexed-file -name=name:bar -new-name=z_Z_42:z_Z_42 -indexed-file=%s -indexed-at=327:60 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK61 %s
// RUN: clang-refactor-test rename-indexed-file -name=bar:part:z_Z_42:part:test -new-name=name:a_200:bar:name:z_Z_42 -indexed-file=%s -indexed-at=327:45 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK62 %s
// RUN: clang-refactor-test rename-indexed-file -name=object:name:object:foo -new-name=z_Z_42:object:usingThing:world -indexed-file=%s -indexed-at=326:30 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK63 %s
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:piece -new-name=struct:piece -indexed-file=%s -indexed-at=336:20 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK64 %s
// RUN: clang-refactor-test rename-indexed-file -name=world:onEntity:object:struct:struct -new-name=withSomething:test:foo:object:bar -indexed-file=%s -indexed-at=335:24 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK65 %s

+(const Object &) object {
  globalArray[12] = [self world: globalArray[i] object: ']' usingThing:  usingThing: globalArray[i]];
  // CHECK66: [[@LINE-1]]:27 -> [[@LINE-1]]:32, [[@LINE-1]]:49 -> [[@LINE-1]]:55, [[@LINE-1]]:61 -> [[@LINE-1]]:71, [[@LINE-1]]:74 -> [[@LINE-1]]:84
  globalArray[12] = ([self.undef_property struct: ']' usingThing: "string"]);
  // CHECK67: [[@LINE-1]]:43 -> [[@LINE-1]]:49, [[@LINE-1]]:55 -> [[@LINE-1]]:65
}
// RUN: clang-refactor-test rename-indexed-file -name=world:object:usingThing:usingThing -new-name=a_200:struct:test:z_Z_42 -indexed-file=%s -indexed-at=353:27 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK66 %s
// RUN: clang-refactor-test rename-indexed-file -name=struct:usingThing -new-name=struct:a_200 -indexed-file=%s -indexed-at=355:43 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK67 %s

-(void) part {
  [self.undef_property withSomething: "string" method: []  {    call() = [self.undef_property object: @"string literal" test: 12
 test: globalArray[i]];
  // CHECK68: [[@LINE-2]]:95 -> [[@LINE-2]]:101, [[@LINE-2]]:121 -> [[@LINE-2]]:125, [[@LINE-1]]:2 -> [[@LINE-1]]:6
  }];
  // CHECK69: [[@LINE-4]]:24 -> [[@LINE-4]]:37, [[@LINE-4]]:48 -> [[@LINE-4]]:54
}
// RUN: clang-refactor-test rename-indexed-file -name=object:test:test -new-name=piece:bar:a_200 -indexed-file=%s -indexed-at=362:95 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK68 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:method -new-name=withSomething:test -indexed-file=%s -indexed-at=362:24 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK69 %s

+(BOOL) method {
  [self object: ^ () { ] } a_200: "]" foo: @{ @1, @3 } piece: []  {    [self.undef_property a_200: 12 foo: 12 withSomething: globalArray[i] onEntity: ']' struct: 12];
  // CHECK70: [[@LINE-1]]:93 -> [[@LINE-1]]:98, [[@LINE-1]]:103 -> [[@LINE-1]]:106, [[@LINE-1]]:111 -> [[@LINE-1]]:124, [[@LINE-1]]:141 -> [[@LINE-1]]:149, [[@LINE-1]]:155 -> [[@LINE-1]]:161
  }
//comment
];
  // CHECK71: [[@LINE-5]]:9 -> [[@LINE-5]]:15, [[@LINE-5]]:28 -> [[@LINE-5]]:33, [[@LINE-5]]:39 -> [[@LINE-5]]:42, [[@LINE-5]]:56 -> [[@LINE-5]]:61
  [super method: 12 struct: ^  {    /*comment*/[super foo: [self bar: @"string literal" method: "string" test: 12 test: "string" world: ']']
 onEntity: globalArray[i] method: ^  {
  }];
  // CHECK72: [[@LINE-3]]:66 -> [[@LINE-3]]:69, [[@LINE-3]]:89 -> [[@LINE-3]]:95, [[@LINE-3]]:106 -> [[@LINE-3]]:110, [[@LINE-3]]:115 -> [[@LINE-3]]:119, [[@LINE-3]]:130 -> [[@LINE-3]]:135
  // CHECK73: [[@LINE-4]]:55 -> [[@LINE-4]]:58, [[@LINE-3]]:2 -> [[@LINE-3]]:10, [[@LINE-3]]:27 -> [[@LINE-3]]:33
  } a_200: "string"];
  // CHECK74: [[@LINE-6]]:10 -> [[@LINE-6]]:16, [[@LINE-6]]:21 -> [[@LINE-6]]:27, [[@LINE-1]]:5 -> [[@LINE-1]]:10
  if (12) {
      [self world: "]" usingThing: @"string literal" + [] () {    call() = [_undef_ivar part: "]" usingThing: 12
];
  // CHECK75: [[@LINE-2]]:89 -> [[@LINE-2]]:93, [[@LINE-2]]:99 -> [[@LINE-2]]:109
  } foo: 12];
  // CHECK76: [[@LINE-4]]:13 -> [[@LINE-4]]:18, [[@LINE-4]]:24 -> [[@LINE-4]]:34, [[@LINE-1]]:5 -> [[@LINE-1]]:8

  }
  [super test: (12) foo: "string" name: []  {
   return @{ @1, @3 };

 } object: "]"];
  // CHECK77: [[@LINE-4]]:10 -> [[@LINE-4]]:14, [[@LINE-4]]:21 -> [[@LINE-4]]:24, [[@LINE-4]]:35 -> [[@LINE-4]]:39, [[@LINE-1]]:4 -> [[@LINE-1]]:10
  return ']';
}
// RUN: clang-refactor-test rename-indexed-file -name=a_200:foo:withSomething:onEntity:struct -new-name=world:bar:class:perform:z_Z_42 -indexed-file=%s -indexed-at=372:93 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK70 %s
// RUN: clang-refactor-test rename-indexed-file -name=object:a_200:foo:piece -new-name=part:withSomething:name:foo -indexed-file=%s -indexed-at=372:9 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK71 %s
// RUN: clang-refactor-test rename-indexed-file -name=bar:method:test:test:world -new-name=withSomething:perform:usingThing:usingThing:method -indexed-file=%s -indexed-at=378:66 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK72 %s
// RUN: clang-refactor-test rename-indexed-file -name=foo:onEntity:method -new-name=foo:name:a_200 -indexed-file=%s -indexed-at=378:55 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK73 %s
// RUN: clang-refactor-test rename-indexed-file -name=method:struct:a_200 -new-name=method:z_Z_42:struct -indexed-file=%s -indexed-at=378:10 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK74 %s
// RUN: clang-refactor-test rename-indexed-file -name=part:usingThing -new-name=withSomething:name -indexed-file=%s -indexed-at=386:89 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK75 %s
// RUN: clang-refactor-test rename-indexed-file -name=world:usingThing:foo -new-name=usingThing:method:class -indexed-file=%s -indexed-at=386:13 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK76 %s
// RUN: clang-refactor-test rename-indexed-file -name=test:foo:name:object -new-name=z_Z_42:world:name:object -indexed-file=%s -indexed-at=393:10 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK77 %s

-(BOOL) object {
  [self part: @{ @1, @3 } foo: "]" method: ([self a_200: @"string literal" test:  part: globalArray[i] object: "]"]) usingThing: @{ @1, @3 }];
  // CHECK78: [[@LINE-1]]:51 -> [[@LINE-1]]:56, [[@LINE-1]]:76 -> [[@LINE-1]]:80, [[@LINE-1]]:83 -> [[@LINE-1]]:87, [[@LINE-1]]:104 -> [[@LINE-1]]:110
  // CHECK79: [[@LINE-2]]:9 -> [[@LINE-2]]:13, [[@LINE-2]]:27 -> [[@LINE-2]]:30, [[@LINE-2]]:36 -> [[@LINE-2]]:42, [[@LINE-2]]:118 -> [[@LINE-2]]:128
  [self onEntity: [] () {    int bar = [self test: "string"/*comment*/ bar: globalArray[i] method: [] () {
  }];
  // CHECK80: [[@LINE-2]]:46 -> [[@LINE-2]]:50, [[@LINE-2]]:72 -> [[@LINE-2]]:75, [[@LINE-2]]:92 -> [[@LINE-2]]:98
  } part: @"string literal" perform: [self.undef_property onEntity: globalArray[i] test: []  {
   if ("]") {
      [super foo: "string" z_Z_42: 12];
  // CHECK81: [[@LINE-1]]:14 -> [[@LINE-1]]:17, [[@LINE-1]]:28 -> [[@LINE-1]]:34

  }

 } method: @"string literal" usingThing: ^ () {
   Object * method = globalArray[i];

 }]];
  // CHECK82: [[@LINE-11]]:59 -> [[@LINE-11]]:67, [[@LINE-11]]:84 -> [[@LINE-11]]:88, [[@LINE-4]]:4 -> [[@LINE-4]]:10, [[@LINE-4]]:30 -> [[@LINE-4]]:40
  // CHECK83: [[@LINE-15]]:9 -> [[@LINE-15]]:17, [[@LINE-12]]:5 -> [[@LINE-12]]:9, [[@LINE-12]]:29 -> [[@LINE-12]]:36
  [self usingThing: ^ () {    int bar = [self onEntity: [self.undef_property perform: "]"
 struct: [self name: "string" withSomething: ^ () {


 }]] foo: ']' object: [_undef_ivar world: [super withSomething: (@{ @1, @3 }) name: @"string literal"
] onEntity: [super perform: "string" class: @"string literal"]] usingThing: 12 == @"string literal"
];
  // CHECK84: [[@LINE-6]]:16 -> [[@LINE-6]]:20, [[@LINE-6]]:31 -> [[@LINE-6]]:44
  // CHECK85: [[@LINE-8]]:78 -> [[@LINE-8]]:85, [[@LINE-7]]:2 -> [[@LINE-7]]:8
  // CHECK86: [[@LINE-5]]:50 -> [[@LINE-5]]:63, [[@LINE-5]]:79 -> [[@LINE-5]]:83
  // CHECK87: [[@LINE-5]]:20 -> [[@LINE-5]]:27, [[@LINE-5]]:38 -> [[@LINE-5]]:43
  // CHECK88: [[@LINE-7]]:36 -> [[@LINE-7]]:41, [[@LINE-6]]:3 -> [[@LINE-6]]:11
  // CHECK89: [[@LINE-12]]:47 -> [[@LINE-12]]:55, [[@LINE-8]]:6 -> [[@LINE-8]]:9, [[@LINE-8]]:15 -> [[@LINE-8]]:21, [[@LINE-7]]:65 -> [[@LINE-7]]:75
  } withSomething: "string" world: "string"];
  // CHECK90: [[@LINE-14]]:9 -> [[@LINE-14]]:19, [[@LINE-1]]:5 -> [[@LINE-1]]:18, [[@LINE-1]]:29 -> [[@LINE-1]]:34
  globalArray[12] = [self.undef_property bar: "string" z_Z_42: ^  {
   [self class: ']' struct: @{ @1, @3 }
 withSomething: ']' class: (']') world: @{ @1, @3 }];
  // CHECK91: [[@LINE-2]]:10 -> [[@LINE-2]]:15, [[@LINE-2]]:21 -> [[@LINE-2]]:27, [[@LINE-1]]:2 -> [[@LINE-1]]:15, [[@LINE-1]]:21 -> [[@LINE-1]]:26, [[@LINE-1]]:34 -> [[@LINE-1]]:39

 }
 a_200: [] () {    ; ;[_undef_ivar class: ']' struct: ^  {
  } z_Z_42: [super withSomething: ^  {


 } foo: globalArray[i] perform: (@"string literal") perform: 12]];
  // CHECK92: [[@LINE-4]]:20 -> [[@LINE-4]]:33, [[@LINE-1]]:4 -> [[@LINE-1]]:7, [[@LINE-1]]:24 -> [[@LINE-1]]:31, [[@LINE-1]]:53 -> [[@LINE-1]]:60
  // CHECK93: [[@LINE-6]]:36 -> [[@LINE-6]]:41, [[@LINE-6]]:47 -> [[@LINE-6]]:53, [[@LINE-5]]:5 -> [[@LINE-5]]:11
  }
];
  // CHECK94: [[@LINE-15]]:42 -> [[@LINE-15]]:45, [[@LINE-15]]:56 -> [[@LINE-15]]:62, [[@LINE-9]]:2 -> [[@LINE-9]]:7
}
// RUN: clang-refactor-test rename-indexed-file -name=a_200:test:part:object -new-name=object:a_200:object:object -indexed-file=%s -indexed-at=410:51 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK78 %s
// RUN: clang-refactor-test rename-indexed-file -name=part:foo:method:usingThing -new-name=world:onEntity:foo:a_200 -indexed-file=%s -indexed-at=410:9 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK79 %s
// RUN: clang-refactor-test rename-indexed-file -name=test:bar:method -new-name=object:method:withSomething -indexed-file=%s -indexed-at=413:46 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK80 %s
// RUN: clang-refactor-test rename-indexed-file -name=foo:z_Z_42 -new-name=foo:class -indexed-file=%s -indexed-at=418:14 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK81 %s
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:test:method:usingThing -new-name=class:a_200:class:object -indexed-file=%s -indexed-at=416:59 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK82 %s
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:part:perform -new-name=name:z_Z_42:class -indexed-file=%s -indexed-at=413:9 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK83 %s
// RUN: clang-refactor-test rename-indexed-file -name=name:withSomething -new-name=withSomething:z_Z_42 -indexed-file=%s -indexed-at=430:16 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK84 %s
// RUN: clang-refactor-test rename-indexed-file -name=perform:struct -new-name=usingThing:struct -indexed-file=%s -indexed-at=429:78 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK85 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:name -new-name=name:foo -indexed-file=%s -indexed-at=433:50 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK86 %s
// RUN: clang-refactor-test rename-indexed-file -name=perform:class -new-name=test:z_Z_42 -indexed-file=%s -indexed-at=434:20 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK87 %s
// RUN: clang-refactor-test rename-indexed-file -name=world:onEntity -new-name=foo:z_Z_42 -indexed-file=%s -indexed-at=433:36 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK88 %s
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:foo:object:usingThing -new-name=test:part:struct:object -indexed-file=%s -indexed-at=429:47 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK89 %s
// RUN: clang-refactor-test rename-indexed-file -name=usingThing:withSomething:world -new-name=a_200:test:struct -indexed-file=%s -indexed-at=429:9 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK90 %s
// RUN: clang-refactor-test rename-indexed-file -name=class:struct:withSomething:class:world -new-name=test:bar:foo:usingThing:usingThing -indexed-file=%s -indexed-at=445:10 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK91 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:foo:perform:perform -new-name=onEntity:struct:piece:withSomething -indexed-file=%s -indexed-at=451:20 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK92 %s
// RUN: clang-refactor-test rename-indexed-file -name=class:struct:z_Z_42 -new-name=onEntity:method:a_200 -indexed-file=%s -indexed-at=450:36 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK93 %s
// RUN: clang-refactor-test rename-indexed-file -name=bar:z_Z_42:a_200 -new-name=withSomething:class:z_Z_42 -indexed-file=%s -indexed-at=444:42 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK94 %s

+(BOOL) world {
  ; ;[self struct: []  {    call() = [self part: 12 class: ^  {


 } test: ^ () {


 } a_200: "string"];
  // CHECK95: [[@LINE-7]]:44 -> [[@LINE-7]]:48, [[@LINE-7]]:53 -> [[@LINE-7]]:58, [[@LINE-4]]:4 -> [[@LINE-4]]:8, [[@LINE-1]]:4 -> [[@LINE-1]]:9
  } piece: "string" z_Z_42: []  {    [self onEntity:  struct: ^ () {
  } world: [_undef_ivar withSomething: globalArray[i] piece: ^  {


 } method:
 part: @"string literal" world: @"string literal"
]];
  // CHECK96: [[@LINE-6]]:25 -> [[@LINE-6]]:38, [[@LINE-6]]:55 -> [[@LINE-6]]:60, [[@LINE-3]]:4 -> [[@LINE-3]]:10, [[@LINE-2]]:2 -> [[@LINE-2]]:6, [[@LINE-2]]:26 -> [[@LINE-2]]:31
  // CHECK97: [[@LINE-8]]:44 -> [[@LINE-8]]:52, [[@LINE-8]]:55 -> [[@LINE-8]]:61, [[@LINE-7]]:5 -> [[@LINE-7]]:10
  }];
  // CHECK98: [[@LINE-18]]:12 -> [[@LINE-18]]:18, [[@LINE-10]]:5 -> [[@LINE-10]]:10, [[@LINE-10]]:21 -> [[@LINE-10]]:27
  if (globalArray[i]) {
      ([_undef_ivar world: ']' class:  usingThing: 12]);
  // CHECK99: [[@LINE-1]]:21 -> [[@LINE-1]]:26, [[@LINE-1]]:32 -> [[@LINE-1]]:37, [[@LINE-1]]:40 -> [[@LINE-1]]:50

  }
  some_type_t foo = [self.undef_property object: ']'
 test: ^ () {
   globalArray[12] = [_undef_ivar z_Z_42: 12 withSomething: [self a_200: ']' perform: globalArray[i] z_Z_42: []  {


 } perform: @"string literal" z_Z_42: globalArray[i]]
 name: 12];
  // CHECK100: [[@LINE-5]]:67 -> [[@LINE-5]]:72, [[@LINE-5]]:78 -> [[@LINE-5]]:85, [[@LINE-5]]:102 -> [[@LINE-5]]:108, [[@LINE-2]]:4 -> [[@LINE-2]]:11, [[@LINE-2]]:31 -> [[@LINE-2]]:37
  // CHECK101: [[@LINE-6]]:35 -> [[@LINE-6]]:41, [[@LINE-6]]:46 -> [[@LINE-6]]:59, [[@LINE-2]]:2 -> [[@LINE-2]]:6

 } bar: "]"
//comment
 usingThing: []  {
   return [self onEntity: 12 < 12 withSomething: []  {


 }];
  // CHECK102: [[@LINE-4]]:17 -> [[@LINE-4]]:25, [[@LINE-4]]:35 -> [[@LINE-4]]:48

 } withSomething: ^  {
   [super class: ([self.undef_property bar: [self withSomething: [self piece: "string" < globalArray[i] onEntity: [self method: 12
 part: "string" usingThing: @{ @1, @3 }]] perform: globalArray[i] bar: []  {
  } name: 12] withSomething: "]"]) world: [] () {
  } usingThing: "]"];
  // CHECK103: [[@LINE-4]]:121 -> [[@LINE-4]]:127, [[@LINE-3]]:2 -> [[@LINE-3]]:6, [[@LINE-3]]:17 -> [[@LINE-3]]:27
  // CHECK104: [[@LINE-5]]:72 -> [[@LINE-5]]:77, [[@LINE-5]]:105 -> [[@LINE-5]]:113
  // CHECK105: [[@LINE-6]]:51 -> [[@LINE-6]]:64, [[@LINE-5]]:43 -> [[@LINE-5]]:50, [[@LINE-5]]:67 -> [[@LINE-5]]:70, [[@LINE-4]]:5 -> [[@LINE-4]]:9
  // CHECK106: [[@LINE-7]]:40 -> [[@LINE-7]]:43, [[@LINE-5]]:15 -> [[@LINE-5]]:28
  // CHECK107: [[@LINE-8]]:11 -> [[@LINE-8]]:16, [[@LINE-6]]:36 -> [[@LINE-6]]:41, [[@LINE-5]]:5 -> [[@LINE-5]]:15

 }];
  // CHECK108: [[@LINE-31]]:42 -> [[@LINE-31]]:48, [[@LINE-30]]:2 -> [[@LINE-30]]:6, [[@LINE-21]]:4 -> [[@LINE-21]]:7, [[@LINE-19]]:2 -> [[@LINE-19]]:12, [[@LINE-12]]:4 -> [[@LINE-12]]:17
}
// RUN: clang-refactor-test rename-indexed-file -name=part:class:test:a_200 -new-name=perform:name:test:struct -indexed-file=%s -indexed-at=480:44 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK95 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:piece:method:part:world -new-name=object:piece:world:piece:world -indexed-file=%s -indexed-at=489:25 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK96 %s
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:struct:world -new-name=onEntity:world:world -indexed-file=%s -indexed-at=488:44 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK97 %s
// RUN: clang-refactor-test rename-indexed-file -name=struct:piece:z_Z_42 -new-name=part:object:onEntity -indexed-file=%s -indexed-at=480:12 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK98 %s
// RUN: clang-refactor-test rename-indexed-file -name=world:class:usingThing -new-name=class:struct:method -indexed-file=%s -indexed-at=500:21 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK99 %s
// RUN: clang-refactor-test rename-indexed-file -name=a_200:perform:z_Z_42:perform:z_Z_42 -new-name=piece:part:piece:class:onEntity -indexed-file=%s -indexed-at=506:67 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK100 %s
// RUN: clang-refactor-test rename-indexed-file -name=z_Z_42:withSomething:name -new-name=usingThing:onEntity:a_200 -indexed-file=%s -indexed-at=506:35 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK101 %s
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:withSomething -new-name=method:usingThing -indexed-file=%s -indexed-at=517:17 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK102 %s
// RUN: clang-refactor-test rename-indexed-file -name=method:part:usingThing -new-name=object:piece:method -indexed-file=%s -indexed-at=524:121 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK103 %s
// RUN: clang-refactor-test rename-indexed-file -name=piece:onEntity -new-name=bar:struct -indexed-file=%s -indexed-at=524:72 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK104 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:perform:bar:name -new-name=usingThing:z_Z_42:class:part -indexed-file=%s -indexed-at=524:51 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK105 %s
// RUN: clang-refactor-test rename-indexed-file -name=bar:withSomething -new-name=world:perform -indexed-file=%s -indexed-at=524:40 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK106 %s
// RUN: clang-refactor-test rename-indexed-file -name=class:world:usingThing -new-name=object:name:name -indexed-file=%s -indexed-at=524:11 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK107 %s
// RUN: clang-refactor-test rename-indexed-file -name=object:test:bar:usingThing:withSomething -new-name=bar:piece:class:perform:class -indexed-file=%s -indexed-at=504:42 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK108 %s

+(Object *) test {
  [super foo: globalArray[i] a_200: globalArray[i]
//comment
 foo: 12
 struct: ^ () {
   [_undef_ivar class: globalArray[i] struct: 12 object: @"string literal" foo: "string"];
  // CHECK109: [[@LINE-1]]:17 -> [[@LINE-1]]:22, [[@LINE-1]]:39 -> [[@LINE-1]]:45, [[@LINE-1]]:50 -> [[@LINE-1]]:56, [[@LINE-1]]:76 -> [[@LINE-1]]:79

 } * [super struct: [self world: ']' struct: []  {
   [self method: []  {


 } withSomething: "string" usingThing: [] () {


 }];
  // CHECK110: [[@LINE-7]]:10 -> [[@LINE-7]]:16, [[@LINE-4]]:4 -> [[@LINE-4]]:17, [[@LINE-4]]:28 -> [[@LINE-4]]:38

 } struct: ^ () {
   return [] () {
  };

 } class: [_undef_ivar part: "string" name: []  {
   int bar = [self withSomething: "string" bar: @"string literal" usingThing: @"string literal"];
  // CHECK111: [[@LINE-1]]:20 -> [[@LINE-1]]:33, [[@LINE-1]]:44 -> [[@LINE-1]]:47, [[@LINE-1]]:67 -> [[@LINE-1]]:77

 }] class: "]"] < 12 foo:  * "]"] + "string" + "]" == [] () {
   some_type_t foo = [_undef_ivar part: 12 name: @"string literal" test: "]" bar: ^  {
  }];
  // CHECK112: [[@LINE-7]]:24 -> [[@LINE-7]]:28, [[@LINE-7]]:39 -> [[@LINE-7]]:43
  // CHECK113: [[@LINE-22]]:27 -> [[@LINE-22]]:32, [[@LINE-22]]:38 -> [[@LINE-22]]:44, [[@LINE-12]]:4 -> [[@LINE-12]]:10, [[@LINE-8]]:4 -> [[@LINE-8]]:9, [[@LINE-4]]:5 -> [[@LINE-4]]:10
  // CHECK114: [[@LINE-23]]:13 -> [[@LINE-23]]:19, [[@LINE-5]]:22 -> [[@LINE-5]]:25
  // CHECK115: [[@LINE-5]]:35 -> [[@LINE-5]]:39, [[@LINE-5]]:44 -> [[@LINE-5]]:48, [[@LINE-5]]:68 -> [[@LINE-5]]:72, [[@LINE-5]]:78 -> [[@LINE-5]]:81

 } < ^  {
   int bar = [self struct: ^ () {


 } a_200: @{ @1, @3 }];
  // CHECK116: [[@LINE-4]]:20 -> [[@LINE-4]]:26, [[@LINE-1]]:4 -> [[@LINE-1]]:9

 } == [super withSomething: []  {    return [_undef_ivar usingThing: [] () {
  }
 world: [] () {


 } foo: 12 perform: globalArray[i] name: @"string literal"];
  // CHECK117: [[@LINE-6]]:58 -> [[@LINE-6]]:68, [[@LINE-4]]:2 -> [[@LINE-4]]:7, [[@LINE-1]]:4 -> [[@LINE-1]]:7, [[@LINE-1]]:12 -> [[@LINE-1]]:19, [[@LINE-1]]:36 -> [[@LINE-1]]:40
  } world: ^  {    // comment
  } object: ^  {    int method = "]";
  }
]];
  // CHECK118: [[@LINE-11]]:14 -> [[@LINE-11]]:27, [[@LINE-4]]:5 -> [[@LINE-4]]:10, [[@LINE-3]]:5 -> [[@LINE-3]]:11
  // CHECK119: [[@LINE-52]]:10 -> [[@LINE-52]]:13, [[@LINE-52]]:30 -> [[@LINE-52]]:35, [[@LINE-50]]:2 -> [[@LINE-50]]:5, [[@LINE-49]]:2 -> [[@LINE-49]]:8
  return []  {    return 12;
  };
  if ([]  {
   some_type_t foo = [_undef_ivar perform: @"string literal" name: globalArray[i] * ']'
 z_Z_42: "]" perform: globalArray[i] class: globalArray[i]
];
  // CHECK120: [[@LINE-3]]:35 -> [[@LINE-3]]:42, [[@LINE-3]]:62 -> [[@LINE-3]]:66, [[@LINE-2]]:2 -> [[@LINE-2]]:8, [[@LINE-2]]:14 -> [[@LINE-2]]:21, [[@LINE-2]]:38 -> [[@LINE-2]]:43

 }) {
      [globalObject message] = [self.undef_property withSomething: ']' test: @"string literal" onEntity: ^  {
   return @{ @1, @3 };

 } onEntity: "string"];
  // CHECK121: [[@LINE-4]]:53 -> [[@LINE-4]]:66, [[@LINE-4]]:72 -> [[@LINE-4]]:76, [[@LINE-4]]:96 -> [[@LINE-4]]:104, [[@LINE-1]]:4 -> [[@LINE-1]]:12

  }
  [globalObject message] = [self onEntity: "string" onEntity: ^  {    call() = [self withSomething: ^ () {
  } == ']' perform: ("string") bar: (']')];
  // CHECK122: [[@LINE-2]]:86 -> [[@LINE-2]]:99, [[@LINE-1]]:12 -> [[@LINE-1]]:19, [[@LINE-1]]:32 -> [[@LINE-1]]:35
  }
];
  // CHECK123: [[@LINE-5]]:34 -> [[@LINE-5]]:42, [[@LINE-5]]:53 -> [[@LINE-5]]:61
  globalArray[12] = [self withSomething: @"string literal" piece: ^  {
   call() = [_undef_ivar perform: (12) bar: ^  {


 } test: "]" name: "string" bar: "]"];
  // CHECK124: [[@LINE-4]]:26 -> [[@LINE-4]]:33, [[@LINE-4]]:40 -> [[@LINE-4]]:43, [[@LINE-1]]:4 -> [[@LINE-1]]:8, [[@LINE-1]]:14 -> [[@LINE-1]]:18, [[@LINE-1]]:29 -> [[@LINE-1]]:32

 } z_Z_42: [] () {    [self onEntity: [] () {


 } piece: "string"];
  // CHECK125: [[@LINE-4]]:29 -> [[@LINE-4]]:37, [[@LINE-1]]:4 -> [[@LINE-1]]:9
  } + @"string literal" object: ^  {
   [globalObject send: [self world: "]" onEntity: ']'
 struct: [] () {
  } perform: [self.undef_property piece: [] () {


 } method: 12 foo: []  {


 } onEntity: "string" * "]" method: "]"] < ']'] other: 42];
  // CHECK126: [[@LINE-7]]:35 -> [[@LINE-7]]:40, [[@LINE-4]]:4 -> [[@LINE-4]]:10, [[@LINE-4]]:15 -> [[@LINE-4]]:18, [[@LINE-1]]:4 -> [[@LINE-1]]:12, [[@LINE-1]]:29 -> [[@LINE-1]]:35
  // CHECK127: [[@LINE-10]]:30 -> [[@LINE-10]]:35, [[@LINE-10]]:41 -> [[@LINE-10]]:49, [[@LINE-9]]:2 -> [[@LINE-9]]:8, [[@LINE-8]]:5 -> [[@LINE-8]]:12

 } method: ^  {    // comment
  }];
  // CHECK128: [[@LINE-27]]:27 -> [[@LINE-27]]:40, [[@LINE-27]]:60 -> [[@LINE-27]]:65, [[@LINE-20]]:4 -> [[@LINE-20]]:10, [[@LINE-15]]:25 -> [[@LINE-15]]:31, [[@LINE-2]]:4 -> [[@LINE-2]]:10
}
// RUN: clang-refactor-test rename-indexed-file -name=class:struct:object:foo -new-name=onEntity:onEntity:struct:foo -indexed-file=%s -indexed-at=557:17 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK109 %s
// RUN: clang-refactor-test rename-indexed-file -name=method:withSomething:usingThing -new-name=object:class:withSomething -indexed-file=%s -indexed-at=561:10 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK110 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:bar:usingThing -new-name=piece:class:test -indexed-file=%s -indexed-at=575:20 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK111 %s
// RUN: clang-refactor-test rename-indexed-file -name=part:name -new-name=bar:object -indexed-file=%s -indexed-at=574:24 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK112 %s
// RUN: clang-refactor-test rename-indexed-file -name=world:struct:struct:class:class -new-name=name:onEntity:z_Z_42:piece:foo -indexed-file=%s -indexed-at=560:27 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK113 %s
// RUN: clang-refactor-test rename-indexed-file -name=struct:foo -new-name=foo:test -indexed-file=%s -indexed-at=560:13 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK114 %s
// RUN: clang-refactor-test rename-indexed-file -name=part:name:test:bar -new-name=class:z_Z_42:onEntity:usingThing -indexed-file=%s -indexed-at=579:35 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK115 %s
// RUN: clang-refactor-test rename-indexed-file -name=struct:a_200 -new-name=usingThing:bar -indexed-file=%s -indexed-at=587:20 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK116 %s
// RUN: clang-refactor-test rename-indexed-file -name=usingThing:world:foo:perform:name -new-name=withSomething:foo:world:test:test -indexed-file=%s -indexed-at=593:58 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK117 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:world:object -new-name=withSomething:class:part -indexed-file=%s -indexed-at=593:14 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK118 %s
// RUN: clang-refactor-test rename-indexed-file -name=foo:a_200:foo:struct -new-name=usingThing:foo:object:test -indexed-file=%s -indexed-at=553:10 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK119 %s
// RUN: clang-refactor-test rename-indexed-file -name=perform:name:z_Z_42:perform:class -new-name=method:usingThing:class:class:world -indexed-file=%s -indexed-at=609:35 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK120 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:test:onEntity:onEntity -new-name=onEntity:z_Z_42:a_200:piece -indexed-file=%s -indexed-at=615:53 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK121 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:perform:bar -new-name=foo:usingThing:world -indexed-file=%s -indexed-at=622:86 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK122 %s
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:onEntity -new-name=foo:perform -indexed-file=%s -indexed-at=622:34 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK123 %s
// RUN: clang-refactor-test rename-indexed-file -name=perform:bar:test:name:bar -new-name=onEntity:withSomething:object:method:test -indexed-file=%s -indexed-at=629:26 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK124 %s
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:piece -new-name=world:z_Z_42 -indexed-file=%s -indexed-at=635:29 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK125 %s
// RUN: clang-refactor-test rename-indexed-file -name=piece:method:foo:onEntity:method -new-name=foo:method:z_Z_42:piece:bar -indexed-file=%s -indexed-at=643:35 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK126 %s
// RUN: clang-refactor-test rename-indexed-file -name=world:onEntity:struct:perform -new-name=test:object:z_Z_42:object -indexed-file=%s -indexed-at=641:30 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK127 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:piece:z_Z_42:object:method -new-name=test:test:withSomething:z_Z_42:name -indexed-file=%s -indexed-at=628:27 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK128 %s

+(int) object {
  [self usingThing: [] () {    ([self.undef_property object: ^  {
  } usingThing: @"string literal"
 test: "string" usingThing: globalArray[i]
 foo: /*]*/ globalArray[i]]);
  // CHECK129: [[@LINE-4]]:54 -> [[@LINE-4]]:60, [[@LINE-3]]:5 -> [[@LINE-3]]:15, [[@LINE-2]]:2 -> [[@LINE-2]]:6, [[@LINE-2]]:17 -> [[@LINE-2]]:27, [[@LINE-1]]:2 -> [[@LINE-1]]:5
  } < ^ () {
   [self z_Z_42:  name: globalArray[i] test: globalArray[i]
 perform: globalArray[i]];
  // CHECK130: [[@LINE-2]]:10 -> [[@LINE-2]]:16, [[@LINE-2]]:19 -> [[@LINE-2]]:23, [[@LINE-2]]:40 -> [[@LINE-2]]:44, [[@LINE-1]]:2 -> [[@LINE-1]]:9

 } perform:  a_200: ([] () {
   [super z_Z_42: "string" z_Z_42: [self perform: [] () {


 } withSomething: ^  {


 }
] world: "]" onEntity: globalArray[i] foo: "]"];
  // CHECK131: [[@LINE-8]]:42 -> [[@LINE-8]]:49, [[@LINE-5]]:4 -> [[@LINE-5]]:17
  // CHECK132: [[@LINE-9]]:11 -> [[@LINE-9]]:17, [[@LINE-9]]:28 -> [[@LINE-9]]:34, [[@LINE-2]]:3 -> [[@LINE-2]]:8, [[@LINE-2]]:14 -> [[@LINE-2]]:22, [[@LINE-2]]:39 -> [[@LINE-2]]:42

 } * "string") a_200: ']'];
  // CHECK133: [[@LINE-23]]:9 -> [[@LINE-23]]:19, [[@LINE-13]]:4 -> [[@LINE-13]]:11, [[@LINE-13]]:14 -> [[@LINE-13]]:19, [[@LINE-1]]:16 -> [[@LINE-1]]:21
  if (12) {
      int bar = [super piece: 12 == globalArray[i] method: [self class: [] () {    return "string";
  }
 class: @"string literal" test: ^ () {
   globalArray[12] = [self class: globalArray[i] z_Z_42: [] () {
  } method: [super test: [] () {
  } struct: @"string literal"
 withSomething: []  {


 } == globalArray[i] class: "]" struct: @"string literal"]];
  // CHECK134: [[@LINE-6]]:20 -> [[@LINE-6]]:24, [[@LINE-5]]:5 -> [[@LINE-5]]:11, [[@LINE-4]]:2 -> [[@LINE-4]]:15, [[@LINE-1]]:22 -> [[@LINE-1]]:27, [[@LINE-1]]:33 -> [[@LINE-1]]:39
  // CHECK135: [[@LINE-8]]:28 -> [[@LINE-8]]:33, [[@LINE-8]]:50 -> [[@LINE-8]]:56, [[@LINE-7]]:5 -> [[@LINE-7]]:11

 } struct: ']' * 12 + @"string literal"]];
  // CHECK136: [[@LINE-14]]:66 -> [[@LINE-14]]:71, [[@LINE-12]]:2 -> [[@LINE-12]]:7, [[@LINE-12]]:27 -> [[@LINE-12]]:31, [[@LINE-1]]:4 -> [[@LINE-1]]:10
  // CHECK137: [[@LINE-15]]:24 -> [[@LINE-15]]:29, [[@LINE-15]]:52 -> [[@LINE-15]]:58

  }
  [globalObject send: [super struct: ^  {
   [globalObject message] = [self.undef_property test: "string" onEntity: "]"] other: 42];
  // CHECK138: [[@LINE-1]]:50 -> [[@LINE-1]]:54, [[@LINE-1]]:65 -> [[@LINE-1]]:73

 } z_Z_42: @"string literal" perform: @"string literal" part: globalArray[i] < 12 struct: [self onEntity: @"string literal" part: "]"]];
  // CHECK139: [[@LINE-1]]:97 -> [[@LINE-1]]:105, [[@LINE-1]]:125 -> [[@LINE-1]]:129
  // CHECK140: [[@LINE-6]]:30 -> [[@LINE-6]]:36, [[@LINE-2]]:4 -> [[@LINE-2]]:10, [[@LINE-2]]:30 -> [[@LINE-2]]:37, [[@LINE-2]]:57 -> [[@LINE-2]]:61, [[@LINE-2]]:83 -> [[@LINE-2]]:89
  if ("]") {
      [super onEntity: ^  {    [self.undef_property withSomething: globalArray[i]
 method: ^ () {
  } onEntity: [self test: [_undef_ivar bar: @{ @1, @3 } < globalArray[i] <  part: ] * "]" method: (([self struct: "string" piece: [] () {
  } struct: "string"])) withSomething: [self a_200: "]" foo: ']' < [self a_200: @"string literal" object: ']' onEntity: "]"] part: 12 usingThing: globalArray[i] name: [self.undef_property perform: @"string literal" world: globalArray[i] method: (12) method: [self.undef_property foo: @"string literal" part: []  {


 } * "string" withSomething: "string"
] perform: @"string literal"]] bar: ']'
]];
  // CHECK141: [[@LINE-7]]:40 -> [[@LINE-7]]:43, [[@LINE-7]]:77 -> [[@LINE-7]]:81
  // CHECK142: [[@LINE-8]]:107 -> [[@LINE-8]]:113, [[@LINE-8]]:124 -> [[@LINE-8]]:129, [[@LINE-7]]:5 -> [[@LINE-7]]:11
  // CHECK143: [[@LINE-8]]:74 -> [[@LINE-8]]:79, [[@LINE-8]]:99 -> [[@LINE-8]]:105, [[@LINE-8]]:111 -> [[@LINE-8]]:119
  // CHECK144: [[@LINE-9]]:280 -> [[@LINE-9]]:283, [[@LINE-9]]:303 -> [[@LINE-9]]:307, [[@LINE-6]]:15 -> [[@LINE-6]]:28
  // CHECK145: [[@LINE-10]]:189 -> [[@LINE-10]]:196, [[@LINE-10]]:216 -> [[@LINE-10]]:221, [[@LINE-10]]:238 -> [[@LINE-10]]:244, [[@LINE-10]]:251 -> [[@LINE-10]]:257, [[@LINE-6]]:3 -> [[@LINE-6]]:10
  // CHECK146: [[@LINE-11]]:46 -> [[@LINE-11]]:51, [[@LINE-11]]:57 -> [[@LINE-11]]:60, [[@LINE-11]]:126 -> [[@LINE-11]]:130, [[@LINE-11]]:135 -> [[@LINE-11]]:145, [[@LINE-11]]:162 -> [[@LINE-11]]:166
  // CHECK147: [[@LINE-13]]:21 -> [[@LINE-13]]:25, [[@LINE-13]]:91 -> [[@LINE-13]]:97, [[@LINE-12]]:25 -> [[@LINE-12]]:38, [[@LINE-8]]:32 -> [[@LINE-8]]:35
  // CHECK148: [[@LINE-16]]:53 -> [[@LINE-16]]:66, [[@LINE-15]]:2 -> [[@LINE-15]]:8, [[@LINE-14]]:5 -> [[@LINE-14]]:13
  } usingThing:
];
  // CHECK149: [[@LINE-19]]:14 -> [[@LINE-19]]:22, [[@LINE-2]]:5 -> [[@LINE-2]]:15

  }
  [self.undef_property test: []  {    if (@"string literal") {
      call() = [self z_Z_42: "string" usingThing: @"string literal"
];
  // CHECK150: [[@LINE-2]]:22 -> [[@LINE-2]]:28, [[@LINE-2]]:39 -> [[@LINE-2]]:49

  }
  } object: globalArray[i]
 test: ^ () {    if ("string" == @"string literal" + ^ () {


 }) {
      int name = 12 * 12;

  }
  } a_200: "]"];
  // CHECK151: [[@LINE-15]]:24 -> [[@LINE-15]]:28, [[@LINE-9]]:5 -> [[@LINE-9]]:11, [[@LINE-8]]:2 -> [[@LINE-8]]:6, [[@LINE-1]]:5 -> [[@LINE-1]]:10
}
// RUN: clang-refactor-test rename-indexed-file -name=object:usingThing:test:usingThing:foo -new-name=part:piece:object:a_200:name -indexed-file=%s -indexed-at=679:54 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK129 %s
// RUN: clang-refactor-test rename-indexed-file -name=z_Z_42:name:test:perform -new-name=a_200:usingThing:usingThing:withSomething -indexed-file=%s -indexed-at=685:10 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK130 %s
// RUN: clang-refactor-test rename-indexed-file -name=perform:withSomething -new-name=piece:struct -indexed-file=%s -indexed-at=690:42 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK131 %s
// RUN: clang-refactor-test rename-indexed-file -name=z_Z_42:z_Z_42:world:onEntity:foo -new-name=withSomething:method:perform:onEntity:bar -indexed-file=%s -indexed-at=690:11 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK132 %s
// RUN: clang-refactor-test rename-indexed-file -name=usingThing:perform:a_200:a_200 -new-name=a_200:world:usingThing:class -indexed-file=%s -indexed-at=679:9 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK133 %s
// RUN: clang-refactor-test rename-indexed-file -name=test:struct:withSomething:class:struct -new-name=withSomething:part:a_200:method:perform -indexed-file=%s -indexed-at=708:20 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK134 %s
// RUN: clang-refactor-test rename-indexed-file -name=class:z_Z_42:method -new-name=part:a_200:part -indexed-file=%s -indexed-at=707:28 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK135 %s
// RUN: clang-refactor-test rename-indexed-file -name=class:class:test:struct -new-name=a_200:world:struct:world -indexed-file=%s -indexed-at=704:66 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK136 %s
// RUN: clang-refactor-test rename-indexed-file -name=piece:method -new-name=name:class -indexed-file=%s -indexed-at=704:24 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK137 %s
// RUN: clang-refactor-test rename-indexed-file -name=test:onEntity -new-name=foo:piece -indexed-file=%s -indexed-at=723:50 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK138 %s
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:part -new-name=class:name -indexed-file=%s -indexed-at=726:97 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK139 %s
// RUN: clang-refactor-test rename-indexed-file -name=struct:z_Z_42:perform:part:struct -new-name=onEntity:part:usingThing:struct:perform -indexed-file=%s -indexed-at=722:30 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK140 %s
// RUN: clang-refactor-test rename-indexed-file -name=bar:part -new-name=part:class -indexed-file=%s -indexed-at=732:40 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK141 %s
// RUN: clang-refactor-test rename-indexed-file -name=struct:piece:struct -new-name=part:onEntity:foo -indexed-file=%s -indexed-at=732:107 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK142 %s
// RUN: clang-refactor-test rename-indexed-file -name=a_200:object:onEntity -new-name=world:bar:onEntity -indexed-file=%s -indexed-at=733:74 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK143 %s
// RUN: clang-refactor-test rename-indexed-file -name=foo:part:withSomething -new-name=usingThing:withSomething:perform -indexed-file=%s -indexed-at=733:280 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK144 %s
// RUN: clang-refactor-test rename-indexed-file -name=perform:world:method:method:perform -new-name=piece:bar:usingThing:class:piece -indexed-file=%s -indexed-at=733:189 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK145 %s
// RUN: clang-refactor-test rename-indexed-file -name=a_200:foo:part:usingThing:name -new-name=z_Z_42:object:name:perform:foo -indexed-file=%s -indexed-at=733:46 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK146 %s
// RUN: clang-refactor-test rename-indexed-file -name=test:method:withSomething:bar -new-name=piece:class:a_200:z_Z_42 -indexed-file=%s -indexed-at=732:21 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK147 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:method:onEntity -new-name=usingThing:foo:object -indexed-file=%s -indexed-at=730:53 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK148 %s
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:usingThing -new-name=perform:piece -indexed-file=%s -indexed-at=730:14 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK149 %s
// RUN: clang-refactor-test rename-indexed-file -name=z_Z_42:usingThing -new-name=test:usingThing -indexed-file=%s -indexed-at=753:22 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK150 %s
// RUN: clang-refactor-test rename-indexed-file -name=test:object:test:a_200 -new-name=foo:world:piece:z_Z_42 -indexed-file=%s -indexed-at=752:24 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK151 %s

+(void) onEntity {
  int perform = [_undef_ivar onEntity: [] () {
   /*comment*/([self z_Z_42: ']' + globalArray[i] a_200:  method: "]" == "]" < ^ () {


 }]);
  // CHECK152: [[@LINE-4]]:22 -> [[@LINE-4]]:28, [[@LINE-4]]:51 -> [[@LINE-4]]:56, [[@LINE-4]]:59 -> [[@LINE-4]]:65

 } method: "]" < globalArray[i] == @"string literal" struct: "string" == @{ @1, @3 } onEntity: @"string literal"
 perform: ^ () {    globalArray[12] = [self foo: ']' * @"string literal" piece: ^  {
  } perform: "string" name: "]"];
  // CHECK153: [[@LINE-2]]:45 -> [[@LINE-2]]:48, [[@LINE-2]]:74 -> [[@LINE-2]]:79, [[@LINE-1]]:5 -> [[@LINE-1]]:12, [[@LINE-1]]:23 -> [[@LINE-1]]:27
  }];
  // CHECK154: [[@LINE-12]]:30 -> [[@LINE-12]]:38, [[@LINE-5]]:4 -> [[@LINE-5]]:10, [[@LINE-5]]:54 -> [[@LINE-5]]:60, [[@LINE-5]]:86 -> [[@LINE-5]]:94, [[@LINE-4]]:2 -> [[@LINE-4]]:9
  return []  {
   [self part: (@"string literal") world: [self struct: [self a_200: "]" test: ']'] piece: "]" withSomething: @"string literal" struct: "]" object: ] bar: ']'];
  // CHECK155: [[@LINE-1]]:63 -> [[@LINE-1]]:68, [[@LINE-1]]:74 -> [[@LINE-1]]:78
  // CHECK156: [[@LINE-2]]:49 -> [[@LINE-2]]:55, [[@LINE-2]]:85 -> [[@LINE-2]]:90, [[@LINE-2]]:96 -> [[@LINE-2]]:109, [[@LINE-2]]:129 -> [[@LINE-2]]:135, [[@LINE-2]]:141 -> [[@LINE-2]]:147
  // CHECK157: [[@LINE-3]]:10 -> [[@LINE-3]]:14, [[@LINE-3]]:36 -> [[@LINE-3]]:41, [[@LINE-3]]:151 -> [[@LINE-3]]:154

 };
}
// RUN: clang-refactor-test rename-indexed-file -name=z_Z_42:a_200:method -new-name=name:world:part -indexed-file=%s -indexed-at=795:22 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK152 %s
// RUN: clang-refactor-test rename-indexed-file -name=foo:piece:perform:name -new-name=name:class:withSomething:method -indexed-file=%s -indexed-at=802:45 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK153 %s
// RUN: clang-refactor-test rename-indexed-file -name=onEntity:method:struct:onEntity:perform -new-name=withSomething:piece:bar:struct:piece -indexed-file=%s -indexed-at=794:30 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK154 %s
// RUN: clang-refactor-test rename-indexed-file -name=a_200:test -new-name=part:z_Z_42 -indexed-file=%s -indexed-at=808:63 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK155 %s
// RUN: clang-refactor-test rename-indexed-file -name=struct:piece:withSomething:struct:object -new-name=foo:name:piece:z_Z_42:bar -indexed-file=%s -indexed-at=808:49 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK156 %s
// RUN: clang-refactor-test rename-indexed-file -name=part:world:bar -new-name=test:a_200:z_Z_42 -indexed-file=%s -indexed-at=808:10 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK157 %s

+(const Object &) foo {
  [self foo: "]" bar: ']' method: ^ () { ] } z_Z_42: [super usingThing: "string" world: [self world: globalArray[i] a_200: []  {    [self z_Z_42: ']' usingThing: ("]")];
  // CHECK158: [[@LINE-1]]:139 -> [[@LINE-1]]:145, [[@LINE-1]]:151 -> [[@LINE-1]]:161
  } foo: (12)] usingThing: []  {
   globalArray[12] = [self name: []  {


 } foo: @"string literal"/*comment*/ withSomething: globalArray[i] test: []  {
  }];
  // CHECK159: [[@LINE-8]]:95 -> [[@LINE-8]]:100, [[@LINE-8]]:117 -> [[@LINE-8]]:122, [[@LINE-6]]:5 -> [[@LINE-6]]:8
  // CHECK160: [[@LINE-6]]:28 -> [[@LINE-6]]:32, [[@LINE-3]]:4 -> [[@LINE-3]]:7, [[@LINE-3]]:38 -> [[@LINE-3]]:51, [[@LINE-3]]:68 -> [[@LINE-3]]:72

 } foo: ^  {    call() = [_undef_ivar world: @"string literal" usingThing: 12 onEntity: @"string literal" struct: ^  {
  }];
  // CHECK161: [[@LINE-2]]:39 -> [[@LINE-2]]:44, [[@LINE-2]]:64 -> [[@LINE-2]]:74, [[@LINE-2]]:79 -> [[@LINE-2]]:87, [[@LINE-2]]:107 -> [[@LINE-2]]:113
  } < "]" test: ^ () {    if ([]  {
  }) {
      call() = [self withSomething: globalArray[i] z_Z_42: globalArray[i]
 foo: 12 onEntity: @"string literal"];
  // CHECK162: [[@LINE-2]]:22 -> [[@LINE-2]]:35, [[@LINE-2]]:52 -> [[@LINE-2]]:58, [[@LINE-1]]:2 -> [[@LINE-1]]:5, [[@LINE-1]]:10 -> [[@LINE-1]]:18

  }
  }]];
  // CHECK163: [[@LINE-22]]:61 -> [[@LINE-22]]:71, [[@LINE-22]]:82 -> [[@LINE-22]]:87, [[@LINE-20]]:16 -> [[@LINE-20]]:26, [[@LINE-11]]:4 -> [[@LINE-11]]:7, [[@LINE-8]]:11 -> [[@LINE-8]]:15
  // CHECK164: [[@LINE-23]]:9 -> [[@LINE-23]]:12, [[@LINE-23]]:18 -> [[@LINE-23]]:21, [[@LINE-23]]:27 -> [[@LINE-23]]:33, [[@LINE-23]]:46 -> [[@LINE-23]]:52
  const Object & struct = ;
  [globalObject message] = [self object: 12
 name: 12 a_200: ^ () {    if ([self object: ^  {
  } foo: "string" * @"string literal"]) {
      [self withSomething: []  {
  }
//comment
 a_200: ^  {
  } foo: "string" piece: [] () {


 }];
  // CHECK165: [[@LINE-10]]:38 -> [[@LINE-10]]:44, [[@LINE-9]]:5 -> [[@LINE-9]]:8
  // CHECK166: [[@LINE-9]]:13 -> [[@LINE-9]]:26, [[@LINE-6]]:2 -> [[@LINE-6]]:7, [[@LINE-5]]:5 -> [[@LINE-5]]:8, [[@LINE-5]]:19 -> [[@LINE-5]]:24

  }
  }];
  // CHECK167: [[@LINE-16]]:34 -> [[@LINE-16]]:40, [[@LINE-15]]:2 -> [[@LINE-15]]:6, [[@LINE-15]]:11 -> [[@LINE-15]]:16
  [self world: @"string literal" withSomething: @{ @1, @3 }];
  // CHECK168: [[@LINE-1]]:9 -> [[@LINE-1]]:14, [[@LINE-1]]:34 -> [[@LINE-1]]:47
}
// RUN: clang-refactor-test rename-indexed-file -name=z_Z_42:usingThing -new-name=method:onEntity -indexed-file=%s -indexed-at=823:139 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK158 %s
// RUN: clang-refactor-test rename-indexed-file -name=world:a_200:foo -new-name=class:test:bar -indexed-file=%s -indexed-at=823:95 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK159 %s
// RUN: clang-refactor-test rename-indexed-file -name=name:foo:withSomething:test -new-name=a_200:perform:piece:method -indexed-file=%s -indexed-at=826:28 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK160 %s
// RUN: clang-refactor-test rename-indexed-file -name=world:usingThing:onEntity:struct -new-name=usingThing:name:onEntity:method -indexed-file=%s -indexed-at=834:39 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK161 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:z_Z_42:foo:onEntity -new-name=perform:struct:bar:object -indexed-file=%s -indexed-at=839:22 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK162 %s
// RUN: clang-refactor-test rename-indexed-file -name=usingThing:world:usingThing:foo:test -new-name=method:onEntity:part:part:bar -indexed-file=%s -indexed-at=823:61 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK163 %s
// RUN: clang-refactor-test rename-indexed-file -name=foo:bar:method:z_Z_42 -new-name=class:onEntity:method:name -indexed-file=%s -indexed-at=823:9 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK164 %s
// RUN: clang-refactor-test rename-indexed-file -name=object:foo -new-name=object:usingThing -indexed-file=%s -indexed-at=849:38 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK165 %s
// RUN: clang-refactor-test rename-indexed-file -name=withSomething:a_200:foo:piece -new-name=class:struct:bar:onEntity -indexed-file=%s -indexed-at=851:13 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK166 %s
// RUN: clang-refactor-test rename-indexed-file -name=object:name:a_200 -new-name=piece:withSomething:withSomething -indexed-file=%s -indexed-at=848:34 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK167 %s
// RUN: clang-refactor-test rename-indexed-file -name=world:withSomething -new-name=part:withSomething -indexed-file=%s -indexed-at=865:9 -indexed-symbol-kind=objc-message %s | FileCheck --check-prefix=CHECK168 %s

+(void) object {
  int test = globalArray[i];
}
