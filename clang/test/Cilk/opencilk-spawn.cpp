// RUN: %clang_cc1 -std=c++1z -fexceptions -fcxx-exceptions -fcilkplus -ftapir=none -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s

int return_stuff(int i);

int return_spawn_test(int i){
  return _Cilk_spawn return_stuff(i);
}

// CHECK-LABEL: define {{(dso_local )?}}i32 @_Z17return_spawn_testi(i32 %i)
// CHECK: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[CONTINUE:.+]] unwind
// CHECK: [[DETACHED]]
// CHECK: %[[CALL:.+]] = invoke i32 @_Z12return_stuffi(i32
// CHECK: to label %[[INVOKECONT:.+]] unwind
// CHECK: [[INVOKECONT]]
// CHECK-NEXT: store i32 %[[CALL]]
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE]]
// CHECK: [[CONTINUE]]
// CHECK-NEXT: sync within %[[SYNCREG]], label %[[SYNCCONT:.+]]
// CHECK: [[SYNCCONT]]
// CHECK-NEXT: %[[RETVALLOAD:.+]] = load i32
// CHECK: ret i32 %[[RETVALLOAD]]

class Bar {
  int val[4] = {0,0,0,0};
public:
  Bar();
  ~Bar();
  Bar(const Bar &that);
  Bar(Bar &&that);
  Bar &operator=(Bar that);
  friend void swap(Bar &left, Bar &right);

  const int getValSpawn(int i) const { return _Cilk_spawn return_stuff(val[i]); }
};

int foo(const Bar &b);

void spawn_infinite_loop() {
  _Cilk_spawn {
  label1: Bar().getValSpawn(0);
  label2: foo(Bar());
    goto label1;
  };
}

// CHECK-LABEL: define {{(dso_local )?}}void @_Z19spawn_infinite_loopv()
// CHECK: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[CONTINUE:.+]] unwind
// CHECK: [[DETACHED]]
// CHECK: br label %[[LABEL1:.+]]
// CHECK: [[LABEL1]]
// CHECK: invoke void @_ZN3BarC1Ev(%class.Bar* %[[REFTMP:.+]])
// CHECK: %[[CALL:.+]] = invoke i32 @_ZNK3Bar11getValSpawnEi(%class.Bar* %[[REFTMP]], i32 0)
// CHECK: call void @_ZN3BarD1Ev(%class.Bar* %[[REFTMP]])
// CHECK: invoke void @_ZN3BarC1Ev(%class.Bar* %[[REFTMP2:.+]])
// CHECK: %[[CALL:.+]] = invoke i32 @_Z3fooRK3Bar(%class.Bar* {{.+}}%[[REFTMP2]])
// CHECK: call void @_ZN3BarD1Ev(%class.Bar* %[[REFTMP2]])
// CHECK-NEXT: br label %[[LABEL1]]
// CHECK: [[CONTINUE]]
// CHECK: ret void

// CHECK-LABEL: define linkonce_odr {{(dso_local )?}}i32 @_ZNK3Bar11getValSpawnEi(%class.Bar
// CHECK: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[CONTINUE:.+]] unwind
// CHECK: [[DETACHED]]
// CHECK: %[[CALL:.+]] = invoke i32 @_Z12return_stuffi(i32
// CHECK: to label %[[INVOKECONT:.+]] unwind
// CHECK: [[INVOKECONT]]
// CHECK-NEXT: store i32 %[[CALL]]
// CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE]]
// CHECK: [[CONTINUE]]
// CHECK-NEXT: sync within %[[SYNCREG]], label %[[SYNCCONT:.+]]
// CHECK: [[SYNCCONT]]
// CHECK-NEXT: %[[RETVALLOAD:.+]] = load i32
// CHECK: ret i32 %[[RETVALLOAD]]
