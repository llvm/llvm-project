// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fcilkplus -ftapir=none -triple x86_64-unknown-linux-gnu -std=c++11 -emit-llvm %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-O0
// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fcilkplus -ftapir=none -triple x86_64-unknown-linux-gnu -std=c++11 -O1 -emit-llvm %s -o - | FileCheck %s --check-prefixes CHECK,CHECK-O1

class Baz {
public:
  Baz();
  ~Baz();
  Baz(const Baz &that);
  Baz(Baz &&that);
  Baz &operator=(Baz that);
  friend void swap(Baz &left, Baz &right);
};

class Bar {
  int val[4] = {0,0,0,0};
public:
  Bar();
  ~Bar();
  Bar(const Bar &that);
  Bar(Bar &&that);
  Bar &operator=(Bar that);
  friend void swap(Bar &left, Bar &right);

  Bar(const Baz &that);

  const int &getVal(int i) const { return val[i]; }
  void incVal(int i) { val[i]++; }
};

class DBar : public Bar {
public:
  DBar();
  ~DBar();
  DBar(const DBar &that);
  DBar(DBar &&that);
  DBar &operator=(DBar that);
  friend void swap(DBar &left, DBar &right);
};

int foo(const Bar &b);

Bar makeBar();
void useBar(Bar b);

DBar makeDBar();
DBar makeDBarFromBar(Bar b);

Baz makeBaz();
Baz makeBazFromBar(Bar b);

void rule_of_four() {
  // CHECK-LABEL: define void @_Z12rule_of_fourv()
  Bar b0;
  Bar b5(_Cilk_spawn makeBar());
  // CHECK: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[CONTINUE:.+]] unwind
  // CHECK: [[DETACHED]]
  // CHECK: invoke void @_Z7makeBarv(%class.Bar* {{(nonnull )?}}sret %[[b5:.+]])
  // CHECK-NEXT: to label %[[REATTACH:.+]] unwind
  // CHECK: [[REATTACH]]
  // CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE]]
  // CHECK: [[CONTINUE]]
  Bar b4 = _Cilk_spawn makeBar();
  // CHECK: detach within %[[SYNCREG]], label %[[DETACHED2:.+]], label %[[CONTINUE2:.+]] unwind
  // CHECK: [[DETACHED2]]
  // CHECK: invoke void @_Z7makeBarv(%class.Bar* {{(nonnull )?}}sret %[[b4:.+]])
  // CHECK-NEXT: to label %[[REATTACH2:.+]] unwind
  // CHECK: [[REATTACH2]]
  // CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE2]]
  // CHECK: [[CONTINUE2]]
  b0 = _Cilk_spawn makeBar();
  // CHECK: detach within %[[SYNCREG]], label %[[DETACHED3:.+]], label %[[CONTINUE3:.+]] unwind
  // CHECK: [[DETACHED3]]
  // CHECK: invoke void @_Z7makeBarv(%class.Bar* {{(nonnull )?}}sret %[[AGGTMP:.+]])
  // CHECK-NEXT: to label %[[INVOKECONT:.+]] unwind
  // CHECK: [[INVOKECONT]]
  // CHECK-NEXT: %[[CALL:.+]] = invoke dereferenceable(16) %class.Bar* @_ZN3BaraSES_(%class.Bar* {{(nonnull )?}}%[[b0:.+]], %class.Bar* {{(nonnull )?}}%[[AGGTMP]])
  // CHECK-NEXT: to label %[[INVOKECONT2:.+]] unwind
  // CHECK: [[INVOKECONT2]]
  // CHECK-NEXT: call void @_ZN3BarD1Ev(%class.Bar* {{(nonnull )?}}%[[AGGTMP]])
  // CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE3]]
  // CHECK: [[CONTINUE3]]
  _Cilk_spawn useBar(b0);
  // CHECK: invoke void @_ZN3BarC1ERKS_(%class.Bar* {{(nonnull )?}}%[[AGGTMP2:.+]], %class.Bar* {{(nonnull )?}}dereferenceable(16) %[[b0:.+]])
  // CHECK-NEXT: to label %[[INVOKECONT3:.+]] unwind
  // CHECK: [[INVOKECONT3]]
  // CHECK: detach within %[[SYNCREG]], label %[[DETACHED4:.+]], label %[[CONTINUE4:.+]] unwind
  // CHECK: [[DETACHED4]]
  // CHECK: invoke void @_Z6useBar3Bar(%class.Bar* {{(nonnull )?}}%[[AGGTMP2]])
  // CHECK-NEXT: to label %[[INVOKECONT4:.+]] unwind
  // CHECK: [[INVOKECONT4]]
  // CHECK-NEXT: call void @_ZN3BarD1Ev(%class.Bar* {{(nonnull )?}}%[[AGGTMP2]])
  // CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE4]]
  // CHECK: [[CONTINUE4]]
}

void derived_class() {
  // CHECK-LABEL: define void @_Z13derived_classv()
  Bar b0, b6, b7;
  Bar b8 = _Cilk_spawn makeDBar(), b2 = _Cilk_spawn makeDBarFromBar(b0);
  // CHECK: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[CONTINUE:.+]] unwind
  // CHECK: [[DETACHED]]
  // CHECK: %[[REFTMP:.+]] = alloca %class.DBar
  // CHECK-O1-NEXT: %[[REFTMPADDR:.+]] = bitcast %class.DBar* %[[REFTMP]] to i8*
  // CHECK-O1-NEXT: call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %[[REFTMPADDR]])
  // CHECK: invoke void @_Z8makeDBarv(%class.DBar* {{(nonnull )?}}sret %[[REFTMP]])
  // CHECK-NEXT: to label %[[INVOKECONT:.+]] unwind
  // CHECK: [[INVOKECONT]]
  // CHECK-O0-NEXT: %[[CAST:.+]] = bitcast %class.DBar* %[[REFTMP]] to %class.Bar*
  // CHECK-O1-NEXT: %[[CAST:.+]] = getelementptr inbounds %class.DBar, %class.DBar* %[[REFTMP]], i64 0, i32 0
  // CHECK-NEXT: invoke void @_ZN3BarC1EOS_(%class.Bar* {{(nonnull )?}}%[[b8:.+]], %class.Bar* {{(nonnull )?}}dereferenceable(16) %[[CAST]])
  // CHECK-NEXT: to label %[[INVOKECONT2:.+]] unwind
  // CHECK: [[INVOKECONT2]]
  // CHECK-NEXT: call void @_ZN4DBarD1Ev(%class.DBar* {{(nonnull )?}}%[[REFTMP]])
  // CHECK-O1-NEXT: call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %[[REFTMPADDR]])
  // CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE]]
  // CHECK: [[CONTINUE]]
  // CHECK: invoke void @_ZN3BarC1ERKS_(%class.Bar* {{(nonnull )?}}%[[AGGTMP:.+]], %class.Bar* {{(nonnull )?}}dereferenceable(16) %[[b0:.+]])
  // CHECK-NEXT: to label %[[INVOKECONT3:.+]] unwind
  // CHECK: [[INVOKECONT3]]
  // CHECK: detach within %[[SYNCREG]], label %[[DETACHED2:.+]], label %[[CONTINUE2:.+]] unwind
  // CHECK: [[DETACHED2]]
  // CHECK: %[[REFTMP2:.+]] = alloca %class.DBar
  // CHECK-O1-NEXT: %[[REFTMP2ADDR:.+]] = bitcast %class.DBar* %[[REFTMP2]] to i8*
  // CHECK-O1-NEXT: call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %[[REFTMP2ADDR]])
  // CHECK: invoke void @_Z15makeDBarFromBar3Bar(%class.DBar* {{(nonnull )?}}sret %[[REFTMP2]], %class.Bar* {{(nonnull )?}}%[[AGGTMP]])
  // CHECK-NEXT: to label %[[INVOKECONT4:.+]] unwind
  // CHECK: [[INVOKECONT4]]
  // CHECK-O0-NEXT: %[[CAST2:.+]] = bitcast %class.DBar* %[[REFTMP2]] to %class.Bar*
  // CHECK-O1-NEXT: %[[CAST2:.+]] = getelementptr inbounds %class.DBar, %class.DBar* %[[REFTMP2]], i64 0, i32 0
  // CHECK-NEXT: invoke void @_ZN3BarC1EOS_(%class.Bar* {{(nonnull )?}}%[[b2:.+]], %class.Bar* {{(nonnull )?}}dereferenceable(16) %[[CAST2]])
  // CHECK-NEXT: to label %[[INVOKECONT5:.+]] unwind
  // CHECK: [[INVOKECONT5]]
  // CHECK-NEXT: call void @_ZN4DBarD1Ev(%class.DBar* {{(nonnull )?}}%[[REFTMP2]])
  // CHECK-O1-NEXT: call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %[[REFTMP2ADDR]])
  // CHECK-NEXT: call void @_ZN3BarD1Ev(%class.Bar* {{(nonnull )?}}%[[AGGTMP]])
  // CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE2]]
  // CHECK: [[CONTINUE2]]
  b6 = _Cilk_spawn makeDBarFromBar(b7);
  // CHECK: invoke void @_ZN3BarC1ERKS_(%class.Bar* {{(nonnull )?}}%[[AGGTMP2:.+]], %class.Bar* {{(nonnull )?}}dereferenceable(16) %[[b7:.+]])
  // CHECK-NEXT: to label %[[INVOKECONT6:.+]] unwind
  // CHECK: [[INVOKECONT6]]
  // CHECK: detach within %[[SYNCREG]], label %[[DETACHED3:.+]], label %[[CONTINUE3:.+]] unwind
  // CHECK: [[DETACHED3]]
  // CHECK: %[[REFTMP3:.+]] = alloca %class.DBar
  // CHECK-O1-NEXT: %[[REFTMP3ADDR:.+]] = bitcast %class.DBar* %[[REFTMP3]] to i8*
  // CHECK-O1-NEXT: call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %[[REFTMP3ADDR]])
  // CHECK: invoke void @_Z15makeDBarFromBar3Bar(%class.DBar* {{(nonnull )?}}sret %[[REFTMP3]], %class.Bar* {{(nonnull )?}}%[[AGGTMP2]])
  // CHECK-NEXT: to label %[[INVOKECONT7:.+]] unwind
  // CHECK: [[INVOKECONT7]]
  // CHECK-O0-NEXT: %[[CAST3:.+]] = bitcast %class.DBar* %[[REFTMP3]] to %class.Bar*
  // CHECK-O1-NEXT: %[[CAST3:.+]] = getelementptr inbounds %class.DBar, %class.DBar* %[[REFTMP3]], i64 0, i32 0
  // CHECK-NEXT: invoke void @_ZN3BarC1EOS_(%class.Bar* {{(nonnull )?}}%[[AGGTMP3:.+]], %class.Bar* {{(nonnull )?}}dereferenceable(16) %[[CAST3]])
  // CHECK-NEXT: to label %[[INVOKECONT8:.+]] unwind
  // CHECK: [[INVOKECONT8]]
  // CHECK-NEXT: %[[CALL:.+]] = invoke dereferenceable(16) %class.Bar* @_ZN3BaraSES_(%class.Bar* {{(nonnull )?}}%[[b6:.+]], %class.Bar* {{(nonnull )?}}%[[AGGTMP3]])
  // CHECK-NEXT: to label %[[INVOKECONT9:.+]] unwind
  // CHECK: [[INVOKECONT9]]
  // CHECK-NEXT: call void @_ZN3BarD1Ev(%class.Bar* {{(nonnull )?}}%[[AGGTMP3]])
  // CHECK-NEXT: call void @_ZN4DBarD1Ev(%class.DBar* {{(nonnull )?}}%[[REFTMP3]])
  // CHECK-O1-NEXT: call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %[[REFTMP3ADDR]])
  // CHECK-NEXT: call void @_ZN3BarD1Ev(%class.Bar* {{(nonnull )?}}%[[AGGTMP2]])
  // CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE3]]
  // CHECK: [[CONTINUE3]]
}

void two_classes() {
  // CHECK-LABEL: define void @_Z11two_classesv()
  Bar b9, b11;
  Bar b12 = _Cilk_spawn makeBaz();
  // CHECK: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[CONTINUE:.+]] unwind
  // CHECK: [[DETACHED]]
  // CHECK: %[[REFTMP:.+]] = alloca %class.Baz
  // CHECK-O1-NEXT: %[[REFTMPADDR:.+]] = getelementptr inbounds %class.Baz, %class.Baz* %[[REFTMP]], i64 0, i32 0
  // CHECK-O1-NEXT: call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %[[REFTMPADDR]])
  // CHECK: invoke void @_Z7makeBazv(%class.Baz* {{(nonnull )?}}sret %[[REFTMP]])
  // CHECK-NEXT: to label %[[INVOKECONT:.+]] unwind
  // CHECK: [[INVOKECONT]]
  // CHECK-NEXT: invoke void @_ZN3BarC1ERK3Baz(%class.Bar* {{(nonnull )?}}%[[b12:.+]], %class.Baz* {{(nonnull )?}}dereferenceable(1) %[[REFTMP]])
  // CHECK-NEXT: to label %[[INVOKECONT2:.+]] unwind
  // CHECK: [[INVOKECONT2]]
  // CHECK-NEXT: call void @_ZN3BazD1Ev(%class.Baz* {{(nonnull )?}}%[[REFTMP]])
  // CHECK-O1-NEXT: call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %[[REFTMPADDR]])
  // CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE]]
  // CHECK: [[CONTINUE]]
  Bar b13 = _Cilk_spawn makeBazFromBar(b9);
  // CHECK: invoke void @_ZN3BarC1ERKS_(%class.Bar* {{(nonnull )?}}%[[AGGTMP:.+]], %class.Bar* {{(nonnull )?}}dereferenceable(16) %[[b9:.+]])
  // CHECK-NEXT: to label %[[INVOKECONT3:.+]] unwind
  // CHECK: [[INVOKECONT3]]
  // CHECK-NEXT: detach within %[[SYNCREG]], label %[[DETACHED2:.+]], label %[[CONTINUE2:.+]] unwind
  // CHECK: [[DETACHED2]]
  // CHECK: %[[REFTMP2:.+]] = alloca %class.Baz
  // CHECK-O1-NEXT: %[[REFTMP2ADDR:.+]] = getelementptr inbounds %class.Baz, %class.Baz* %[[REFTMP2]], i64 0, i32 0
  // CHECK-O1-NEXT: call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %[[REFTMP2ADDR]])
  // CHECK: invoke void @_Z14makeBazFromBar3Bar(%class.Baz* {{(nonnull )?}}sret %[[REFTMP2]], %class.Bar* {{(nonnull )?}}%[[AGGTMP]])
  // CHECK-NEXT: to label %[[INVOKECONT4:.+]] unwind
  // CHECK: [[INVOKECONT4]]
  // CHECK-NEXT: invoke void @_ZN3BarC1ERK3Baz(%class.Bar* {{(nonnull )?}}%[[b13:.+]], %class.Baz* {{(nonnull )?}}dereferenceable(1) %[[REFTMP2]])
  // CHECK-NEXT: to label %[[INVOKECONT5:.+]] unwind
  // CHECK: [[INVOKECONT5]]
  // CHECK-NEXT: call void @_ZN3BazD1Ev(%class.Baz* {{(nonnull )?}}%[[REFTMP2]])
  // CHECK-O1-NEXT: call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %[[REFTMP2ADDR]])
  // CHECK-NEXT: call void @_ZN3BarD1Ev(%class.Bar* {{(nonnull )?}}%[[AGGTMP]])
  // CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE2]]
  // CHECK: [[CONTINUE2]]
  b9 = _Cilk_spawn makeBazFromBar(b11);
  // CHECK: invoke void @_ZN3BarC1ERKS_(%class.Bar* {{(nonnull )?}}%[[AGGTMP2:.+]], %class.Bar* {{(nonnull )?}}dereferenceable(16) %[[b11:.+]])
  // CHECK-NEXT: to label %[[INVOKECONT6:.+]] unwind
  // CHECK: [[INVOKECONT6]]
  // CHECK-NEXT: detach within %[[SYNCREG]], label %[[DETACHED3:.+]], label %[[CONTINUE3:.+]] unwind
  // CHECK: [[DETACHED3]]
  // CHECK: %[[REFTMP3:.+]] = alloca %class.Baz
  // CHECK-O1-NEXT: %[[REFTMP3ADDR:.+]] = getelementptr inbounds %class.Baz, %class.Baz* %[[REFTMP3]], i64 0, i32 0
  // CHECK-O1-NEXT: call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %[[REFTMP3ADDR]])
  // CHECK: invoke void @_Z14makeBazFromBar3Bar(%class.Baz* {{(nonnull )?}}sret %[[REFTMP3]], %class.Bar* {{(nonnull )?}}%[[AGGTMP2]])
  // CHECK-NEXT: to label %[[INVOKECONT7:.+]] unwind
  // CHECK: [[INVOKECONT7]]
  // CHECK-NEXT: invoke void @_ZN3BarC1ERK3Baz(%class.Bar* {{(nonnull )?}}%[[AGGTMP3:.+]], %class.Baz* {{(nonnull )?}}dereferenceable(1) %[[REFTMP3]])
  // CHECK-NEXT: to label %[[INVOKECONT8:.+]] unwind
  // CHECK: [[INVOKECONT8]]
  // CHECK-NEXT: %[[CALL:.+]] = invoke dereferenceable(16) %class.Bar* @_ZN3BaraSES_(%class.Bar* {{(nonnull )?}}%[[b9:.+]], %class.Bar* {{(nonnull )?}}%[[AGGTMP3]])
  // CHECK-NEXT: to label %[[INVOKECONT9:.+]] unwind
  // CHECK: [[INVOKECONT9]]
  // CHECK-NEXT: call void @_ZN3BarD1Ev(%class.Bar* {{(nonnull )?}}%[[AGGTMP3]])
  // CHECK-NEXT: call void @_ZN3BazD1Ev(%class.Baz* {{(nonnull )?}}%[[REFTMP3]])
  // CHECK-O1-NEXT: call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %[[REFTMP3ADDR]])
  // CHECK-NEXT: call void @_ZN3BarD1Ev(%class.Bar* {{(nonnull )?}}%[[AGGTMP2]])
  // CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE3]]
  // CHECK: [[CONTINUE3]]
}

void array_out() {
  // CHECK-LABEL: define void @_Z9array_outv()
  // int Arri[5];
  // Example that produces a BinAssign expr.
  // bool Assign0 = (Arri[0] = foo(makeBazFromBar((Bar()))));
  // Pretty sure the following just isn't legal Cilk.
  // bool Assign1 = (Arri[1] = _Cilk_spawn foo(makeBazFromBar((Bar()))));

  Bar ArrBar[5];
  // ArrBar[0] = makeBazFromBar((Bar()));
  ArrBar[1] = _Cilk_spawn makeBazFromBar((Bar()));
  // CHECK: %[[ARRIDX:.+]] = getelementptr inbounds [5 x %class.Bar], [5 x %class.Bar]* %[[ArrBar:.+]], i64 0, i64 1
  // CHECK: invoke void @_ZN3BarC1Ev(%class.Bar* {{(nonnull )?}}%[[AGGTMP:.+]])
  // CHECK-NEXT: to label %[[INVOKECONT:.+]] unwind
  // CHECK: [[INVOKECONT]]
  // CHECK-NEXT: detach within %[[SYNCREG:.+]], label %[[DETACHED:.+]], label %[[CONTINUE:.+]] unwind
  // CHECK: [[DETACHED]]
  // CHECK: %[[REFTMP:.+]] = alloca %class.Baz
  // CHECK-O1-NEXT: %[[REFTMPADDR:.+]] = getelementptr inbounds %class.Baz, %class.Baz* %[[REFTMP]], i64 0, i32 0
  // CHECK-O1-NEXT: call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %[[REFTMPADDR]])
  // CHECK: invoke void @_Z14makeBazFromBar3Bar(%class.Baz* {{(nonnull )?}}sret %[[REFTMP]], %class.Bar* {{(nonnull )?}}%[[AGGTMP]])
  // CHECK: to label %[[INVOKECONT2:.+]] unwind
  // CHECK: [[INVOKECONT2]]
  // CHECK-NEXT: invoke void @_ZN3BarC1ERK3Baz(%class.Bar* {{(nonnull )?}}%[[AGGTMP2:.+]], %class.Baz* {{(nonnull )?}}dereferenceable(1) %[[REFTMP]])
  // CHECK-NEXT: to label %[[INVOKECONT3:.+]] unwind
  // CHECK: [[INVOKECONT3]]
  // CHECK-NEXT: %[[CALL:.+]] = invoke dereferenceable(16) %class.Bar* @_ZN3BaraSES_(%class.Bar* {{(nonnull )?}}%[[ARRIDX]], %class.Bar* {{(nonnull )?}}%[[AGGTMP2]])
  // CHECK-NEXT: to label %[[INVOKECONT4:.+]] unwind
  // CHECK: [[INVOKECONT4]]
  // CHECK-NEXT: call void @_ZN3BarD1Ev(%class.Bar* {{(nonnull )?}}%[[AGGTMP2]])
  // CHECK-NEXT: call void @_ZN3BazD1Ev(%class.Baz* {{(nonnull )?}}%[[REFTMP]])
  // CHECK-O1-NEXT: call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %[[REFTMPADDR]])
  // CHECK-NEXT: call void @_ZN3BarD1Ev(%class.Bar* {{(nonnull )?}}%[[AGGTMP]])
  // CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE]]
  // CHECL: [[CONTINUE]]

  // List initialization
  // Bar ListBar1[3] = { Bar(), makeBar(), makeBazFromBar((Bar())) };
  Bar ListBar2[3] = { _Cilk_spawn Bar(), _Cilk_spawn makeBar(), _Cilk_spawn makeBazFromBar((Bar())) };
  // CHECK: %[[ARRIDX2:.+]] = getelementptr inbounds [3 x %class.Bar], [3 x %class.Bar]* %[[ListBar2:.+]], i64 0, i64 0
  // CHECK: detach within %[[SYNCREG]], label %[[DETACHED2:.+]], label %[[CONTINUE2:.+]] unwind
  // CHECK: [[DETACHED2]]
  // CHECK: invoke void @_ZN3BarC1Ev(%class.Bar* {{(nonnull )?}}%[[ARRIDX2]])
  // CHECK-NEXT: to label %[[INVOKECONT5:.+]] unwind
  // CHECK: [[INVOKECONT5]]
  // CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE2]]
  // CHECK: [[CONTINUE2]]

  // CHECK-O0: %[[ARRIDX3:.+]] = getelementptr inbounds %class.Bar, %class.Bar* %[[ARRIDX2]], i64 1
  // CHECK-O1: %[[ARRIDX3:.+]] = getelementptr inbounds [3 x %class.Bar], [3 x %class.Bar]* %[[ListBar2]], i64 0, i64 1
  // CHECK: detach within %[[SYNCREG]], label %[[DETACHED3:.+]], label %[[CONTINUE3:.+]] unwind
  // CHECK: [[DETACHED3]]
  // CHECK: invoke void @_Z7makeBarv(%class.Bar* {{(nonnull )?}}sret %[[ARRIDX3]])
  // CHECK-NEXT: to label %[[INVOKECONT6:.+]] unwind
  // CHECK: [[INVOKECONT6]]
  // CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE3]]

  // CHECK-O0: %[[ARRIDX4:.+]] = getelementptr inbounds %class.Bar, %class.Bar* %[[ARRIDX3]], i64 1
  // CHECK-O1: %[[ARRIDX4:.+]] = getelementptr inbounds [3 x %class.Bar], [3 x %class.Bar]* %[[ListBar2]], i64 0, i64 2
  // CHECK: invoke void @_ZN3BarC1Ev(%class.Bar* {{(nonnull )?}}%[[AGGTMP3:.+]])
  // CHECK-NEXT: to label %[[INVOKECONT7:.+]] unwind
  // CHECK: [[INVOKECONT7]]
  // CHECK-NEXT: detach within %[[SYNCREG]], label %[[DETACHED4:.+]], label %[[CONTINUE4:.+]] unwind
  // CHECK: [[DETACHED4]]
  // CHECK: %[[REFTMP2:.+]] = alloca %class.Baz
  // CHECK-O1-NEXT: %[[REFTMP2ADDR:.+]] = getelementptr inbounds %class.Baz, %class.Baz* %[[REFTMP2]], i64 0, i32 0
  // CHECK-O1-NEXT: call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %[[REFTMP2ADDR]])
  // CHECK: invoke void @_Z14makeBazFromBar3Bar(%class.Baz* {{(nonnull )?}}sret %[[REFTMP2]], %class.Bar* {{(nonnull )?}}%[[AGGTMP3]])
  // CHECK: to label %[[INVOKECONT8:.+]] unwind
  // CHECK: [[INVOKECONT8]]
  // CHECK-NEXT: invoke void @_ZN3BarC1ERK3Baz(%class.Bar* {{(nonnull )?}}%[[ARRIDX4:.+]], %class.Baz* {{(nonnull )?}}dereferenceable(1) %[[REFTMP2]])
  // CHECK-NEXT: to label %[[INVOKECONT9:.+]] unwind
  // CHECK: [[INVOKECONT9]]
  // CHECK-NEXT: call void @_ZN3BazD1Ev(%class.Baz* {{(nonnull )?}}%[[REFTMP2]])
  // CHECK-O1-NEXT: call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %[[REFTMP2ADDR]])
  // CHECK-NEXT: call void @_ZN3BarD1Ev(%class.Bar* {{(nonnull )?}}%[[AGGTMP3]])
  // CHECK-NEXT: reattach within %[[SYNCREG]], label %[[CONTINUE4]]
  // CHECK: [[CONTINUE4]]
}
