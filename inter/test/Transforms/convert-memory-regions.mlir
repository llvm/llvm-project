// RUN: inter-opt %s --inter-convert-memory | FileCheck %s

// CHECK-LABEL: func.func @if_thread
// CHECK: [[ENTRY:%.*]] = xw.token
// CHECK: [[PRE:%.*]] = xw.store {{.*}} dep [[ENTRY]]
// CHECK: [[IF:%.*]]:2 = scf.if {{.*}} -> (i32, !xemachine.mem.token) {
// CHECK-NOT: xw.token
// CHECK: {{%.*}}, [[THEN:%.*]] = xw.load {{.*}} dep [[PRE]]
// CHECK: scf.yield {{%.*}}, [[THEN]] : i32, !xemachine.mem.token
// CHECK: } else {
// CHECK: scf.yield {{%.*}}, [[PRE]] : i32, !xemachine.mem.token
// CHECK: }
// CHECK: {{%.*}}, {{%.*}} = xw.load {{.*}} dep [[IF]]#1
func.func @if_thread(%pointer: !llvm.ptr<1>, %condition: i1, %index: i64,
                     %value: i32) attributes {xemachine.kernel} {
  %address = llvm.getelementptr %pointer[%index]
      : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
  llvm.store %value, %address : i32, !llvm.ptr<1>
  %result = scf.if %condition -> i32 {
    %loaded = llvm.load %address : !llvm.ptr<1> -> i32
    scf.yield %loaded : i32
  } else {
    scf.yield %value : i32
  }
  %after = llvm.load %address : !llvm.ptr<1> -> i32
  return
}

// Nested branches must pass the inner result through the enclosing region.
// CHECK-LABEL: func.func @nested_if
// CHECK: [[ENTRY:%.*]] = xw.token
// CHECK: [[OUTER:%.*]] = scf.if
// CHECK: [[INNER:%.*]] = scf.if
// CHECK: [[WRITE:%.*]] = xw.store {{.*}} dep [[ENTRY]]
// CHECK: scf.yield [[WRITE]] : !xemachine.mem.token
// CHECK: } else {
// CHECK: scf.yield [[ENTRY]] : !xemachine.mem.token
// CHECK: }
// CHECK: scf.yield [[INNER]] : !xemachine.mem.token
// CHECK: } else {
// CHECK: {{%.*}}, [[ELSE:%.*]] = xw.load {{.*}} dep [[ENTRY]]
// CHECK: scf.yield [[ELSE]] : !xemachine.mem.token
// CHECK: }
// CHECK: {{%.*}}, {{%.*}} = xw.load {{.*}} dep [[OUTER]]
func.func @nested_if(%pointer: !llvm.ptr<1>, %outer_condition: i1,
                     %inner_condition: i1, %index: i64, %value: i32)
    attributes {xemachine.kernel} {
  %address = llvm.getelementptr %pointer[%index]
      : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
  scf.if %outer_condition {
    scf.if %inner_condition {
      llvm.store %value, %address : i32, !llvm.ptr<1>
    } else {
    }
  } else {
    %loaded = llvm.load %address : !llvm.ptr<1> -> i32
  }
  %after = llvm.load %address : !llvm.ptr<1> -> i32
  return
}

// This exercises another RegionBranchOpInterface implementation without any
// corresponding operation-specific handling in the conversion pass.
// CHECK-LABEL: func.func @index_switch
// CHECK: [[ENTRY:%.*]] = xw.token
// CHECK: [[SWITCH:%.*]]:2 = scf.index_switch {{.*}} -> i32, !xemachine.mem.token
// CHECK: case 0 {
// CHECK: {{%.*}}, [[CASE:%.*]] = xw.load {{.*}} dep [[ENTRY]]
// CHECK: scf.yield {{%.*}}, [[CASE]] : i32, !xemachine.mem.token
// CHECK: default {
// CHECK: scf.yield {{%.*}}, [[ENTRY]] : i32, !xemachine.mem.token
// CHECK: {{%.*}}, {{%.*}} = xw.load {{.*}} dep [[SWITCH]]#1
func.func @index_switch(%pointer: !llvm.ptr<1>, %selector: index,
                        %index: i64, %value: i32)
    attributes {xemachine.kernel} {
  %address = llvm.getelementptr %pointer[%index]
      : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
  %result = scf.index_switch %selector -> i32
  case 0 {
    %loaded = llvm.load %address : !llvm.ptr<1> -> i32
    scf.yield %loaded : i32
  }
  default {
    scf.yield %value : i32
  }
  %after = llvm.load %address : !llvm.ptr<1> -> i32
  return
}

// Loop-like region branches use the same interface path as all other region
// branches: the token is an init, backedge value, and result.
// CHECK-LABEL: func.func @for_thread
// CHECK: [[ENTRY:%.*]] = xw.token
// CHECK: [[PRE:%.*]] = xw.store {{.*}} dep [[ENTRY]]
// CHECK: [[LOOP:%.*]]:2 = scf.for {{.*}} iter_args({{%.*}} = {{%.*}}, [[ITER:%.*]] = [[PRE]])
// CHECK-SAME: -> (i32, !xemachine.mem.token) {
// CHECK-NOT: xw.token
// CHECK: {{%.*}}, [[READ:%.*]] = xw.load {{.*}} dep [[ITER]]
// CHECK: [[WRITE:%.*]] = xw.store {{.*}} dep [[READ]]
// CHECK: scf.yield {{%.*}}, [[WRITE]] : i32, !xemachine.mem.token
// CHECK: }
// CHECK: {{%.*}}, {{%.*}} = xw.load {{.*}} dep [[LOOP]]#1
func.func @for_thread(%pointer: !llvm.ptr<1>, %lower: index, %upper: index,
                      %step: index, %index: i64, %value: i32)
    attributes {xemachine.kernel} {
  %address = llvm.getelementptr %pointer[%index]
      : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
  llvm.store %value, %address : i32, !llvm.ptr<1>
  %result = scf.for %iv = %lower to %upper step %step
      iter_args(%iter = %value) -> i32 {
    %loaded = llvm.load %address : !llvm.ptr<1> -> i32
    llvm.store %loaded, %address : i32, !llvm.ptr<1>
    scf.yield %loaded : i32
  }
  %after = llvm.load %address : !llvm.ptr<1> -> i32
  return
}

// The token is an init/before argument, a condition/after argument, a
// backedge yield, and a loop result. This covers both scf.while value circuits.
// CHECK-LABEL: func.func @while_thread
// CHECK: [[ENTRY:%.*]] = xw.token
// CHECK: [[PRE:%.*]] = xw.store {{.*}} dep [[ENTRY]]
// CHECK: [[LOOP:%.*]]:2 = scf.while ({{.*}}, [[BEFORE:%.*]] = [[PRE]])
// CHECK-SAME: : (i32, !xemachine.mem.token) -> (i32, !xemachine.mem.token) {
// CHECK-NOT: xw.token
// CHECK: {{%.*}}, [[CONDITION_TOKEN:%.*]] = xw.load {{.*}} dep [[BEFORE]]
// CHECK: scf.condition({{%.*}}) {{%.*}}, [[CONDITION_TOKEN]] : i32, !xemachine.mem.token
// CHECK: } do {
// CHECK: ^{{.*}}({{%.*}}: i32, [[AFTER:%.*]]: !xemachine.mem.token):
// CHECK: [[BODY:%.*]] = xw.store {{.*}} dep [[AFTER]]
// CHECK: scf.yield {{%.*}}, [[BODY]] : i32, !xemachine.mem.token
// CHECK: }
// CHECK: {{%.*}}, {{%.*}} = xw.load {{.*}} dep [[LOOP]]#1
func.func @while_thread(%pointer: !llvm.ptr<1>, %condition: i1, %index: i64,
                        %value: i32) attributes {xemachine.kernel} {
  %address = llvm.getelementptr %pointer[%index]
      : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
  llvm.store %value, %address : i32, !llvm.ptr<1>
  %result = scf.while (%before = %value) : (i32) -> i32 {
    %loaded = llvm.load %address : !llvm.ptr<1> -> i32
    scf.condition(%condition) %loaded : i32
  } do {
  ^bb0(%after: i32):
    llvm.store %after, %address : i32, !llvm.ptr<1>
    scf.yield %after : i32
  }
  %after = llvm.load %address : !llvm.ptr<1> -> i32
  return
}

// CHECK-LABEL: func.func @while_empty_signature
// CHECK: [[ENTRY:%.*]] = xw.token
// CHECK: [[LOOP:%.*]] = scf.while ([[BEFORE:%.*]] = [[ENTRY]])
// CHECK-SAME: : (!xemachine.mem.token) -> !xemachine.mem.token {
// CHECK: {{%.*}}, [[CONDITION_TOKEN:%.*]] = xw.load {{.*}} dep [[BEFORE]]
// CHECK: scf.condition({{%.*}}) [[CONDITION_TOKEN]] : !xemachine.mem.token
// CHECK: } do {
// CHECK: ^{{.*}}([[AFTER:%.*]]: !xemachine.mem.token):
// CHECK: [[BODY:%.*]] = xw.store {{.*}} dep [[AFTER]]
// CHECK: scf.yield [[BODY]] : !xemachine.mem.token
// CHECK: }
// CHECK: {{%.*}}, {{%.*}} = xw.load {{.*}} dep [[LOOP]]
func.func @while_empty_signature(%pointer: !llvm.ptr<1>, %condition: i1,
                                 %index: i64, %value: i32)
    attributes {xemachine.kernel} {
  %address = llvm.getelementptr %pointer[%index]
      : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
  scf.while : () -> () {
    %loaded = llvm.load %address : !llvm.ptr<1> -> i32
    scf.condition(%condition)
  } do {
    llvm.store %value, %address : i32, !llvm.ptr<1>
    scf.yield
  }
  %after = llvm.load %address : !llvm.ptr<1> -> i32
  return
}
