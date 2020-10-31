// RUN: %clang_cc1 -O1 -disable-llvm-passes -emit-llvm %s -o - -triple=x86_64-linux-gnu | FileCheck %s

extern volatile int i;

void OneCaseL() {
  // CHECK-LABEL: define{{.*}}OneCaseL
  // CHECK: switch
  // CHECK: {{.*}} !prof !6
  switch (i) {
    [[likely]] case 1: break;
  }
}

void OneCaseU() {
  // CHECK-LABEL: define{{.*}}OneCaseU
  // CHECK: switch
  // CHECK: {{.*}} !prof !7
  switch (i) {
    [[unlikely]] case 1: ++i; break;
  }
}

void TwoCasesLN() {
  // CHECK-LABEL: define{{.*}}TwoCasesLN
  // CHECK: switch
  // CHECK: {{.*}} !prof !8
  switch (i) {
    [[likely]] case 1: break;
    case 2: break;
  }
}

void TwoCasesUN() {
  // CHECK-LABEL: define{{.*}}TwoCasesUN
  // CHECK: switch
  // CHECK: {{.*}} !prof !9
  switch (i) {
    [[unlikely]] case 1: break;
    case 2: break;
  }
}

void TwoCasesLU() {
  // CHECK-LABEL: define{{.*}}TwoCasesLU
  // CHECK: switch
  // CHECK: {{.*}} !prof !10
  switch (i) {
    [[likely]] case 1: break;
    [[unlikely]] case 2: break;
  }
}

void CasesFallthroughNNLN() {
  // CHECK-LABEL: define{{.*}}CasesFallthroughNNLN
  // CHECK: switch
  // CHECK: {{.*}} !prof !11
  switch (i) {
    case 1:
    case 2:
    [[likely]] case 3:
    case 4: break;
  }
}

void CasesFallthroughNNUN() {
  // CHECK-LABEL: define{{.*}}CasesFallthroughNNUN
  // CHECK: switch
  // CHECK: {{.*}} !prof !12
  switch (i) {
    case 1:
    case 2:
    [[unlikely]] case 3:
    case 4: break;
  }
}

void CasesFallthroughRangeSmallLN() {
  // CHECK-LABEL: define{{.*}}CasesFallthroughRangeSmallLN
  // CHECK: switch
  // CHECK: {{.*}} !prof !13
  switch (i) {
    case 1 ... 5: ++i;
    case 102:
    [[likely]] case 103:
    case 104: break;
  }
}

void CasesFallthroughRangeSmallUN() {
  // CHECK-LABEL: define{{.*}}CasesFallthroughRangeSmallUN
  // CHECK: switch
  // CHECK: {{.*}} !prof !14
  switch (i) {
    case 1 ... 5: ++i;
    case 102:
    [[unlikely]] case 103:
    case 104: break;
  }
}

void CasesFallthroughRangeLargeLLN() {
  // CHECK-LABEL: define{{.*}}CasesFallthroughRangeLargeLLN
  // CHECK: switch
  // CHECK: {{.*}} !prof !8
  // CHECK: caserange
  // CHECK: br{{.*}} !prof !15
  switch (i) {
    [[likely]] case 0 ... 64:
    [[likely]] case 1003:
    case 104: break;
  }
}

void CasesFallthroughRangeLargeUUN() {
  // CHECK-LABEL: define{{.*}}CasesFallthroughRangeLargeUUN
  // CHECK: switch
  // CHECK: {{.*}} !prof !9
  // CHECK: caserange
  // CHECK: br{{.*}} !prof !16
  switch (i) {
    [[unlikely]] case 0 ... 64:
    [[unlikely]] case 1003:
    case 104: break;
  }
}

void OneCaseDefaultL() {
  // CHECK-LABEL: define{{.*}}OneCaseDefaultL
  // CHECK: switch
  // CHECK: {{.*}} !prof !17
  switch (i) {
    case 1: break;
    [[likely]] default: break;
  }
}

void OneCaseDefaultU() {
  // CHECK-LABEL: define{{.*}}OneCaseDefaultU
  // CHECK: switch
  // CHECK: {{.*}} !prof !18
  switch (i) {
    case 1: break;
    [[unlikely]] default: break;
  }
}

void TwoCasesDefaultLNL() {
  // CHECK-LABEL: define{{.*}}TwoCasesDefaultLNL
  // CHECK: switch
  // CHECK: {{.*}} !prof !19
  switch (i) {
    [[likely]] case 1: break;
    case 2: break;
    [[likely]] default: break;
  }
}

void TwoCasesDefaultLNN() {
  // CHECK-LABEL: define{{.*}}TwoCasesDefaultLNN
  // CHECK: switch
  // CHECK: {{.*}} !prof !8
  switch (i) {
    [[likely]] case 1: break;
    case 2: break;
    default: break;
  }
}

void TwoCasesDefaultLNU() {
  // CHECK-LABEL: define{{.*}}TwoCasesDefaultLNU
  // CHECK: switch
  // CHECK: {{.*}} !prof !20
  switch (i) {
    [[likely]] case 1: break;
    case 2: break;
    [[unlikely]] default: break;
  }
}

// CHECK: !6 = !{!"branch_weights", i32 357913942, i32 715827883}
// CHECK: !7 = !{!"branch_weights", i32 536870912, i32 1}
// CHECK: !8 = !{!"branch_weights", i32 238609295, i32 715827883, i32 238609295}
// CHECK: !9 = !{!"branch_weights", i32 357913942, i32 1, i32 357913942}
// CHECK: !10 = !{!"branch_weights", i32 357913942, i32 715827883, i32 1}
// CHECK: !11 = !{!"branch_weights", i32 143165577, i32 143165577, i32 143165577, i32 715827883, i32 143165577}
// CHECK: !12 = !{!"branch_weights", i32 214748365, i32 214748365, i32 214748365, i32 1, i32 214748365}
// CHECK: !13 = !{!"branch_weights", i32 79536432, i32 79536432, i32 79536432, i32 79536432, i32 79536432, i32 79536432, i32 79536432, i32 715827883, i32 79536432}
// CHECK: !14 = !{!"branch_weights", i32 119304648, i32 119304648, i32 119304648, i32 119304648, i32 119304648, i32 119304648, i32 119304648, i32 1, i32 119304648}
// CHECK: !15 = !{!"branch_weights", i32 2000, i32 1}
// CHECK: !16 = !{!"branch_weights", i32 1, i32 2000}
// CHECK: !17 = !{!"branch_weights", i32 715827883, i32 357913942}
// CHECK: !18 = !{!"branch_weights", i32 1, i32 536870912}
// CHECK: !19 = !{!"branch_weights", i32 536870912, i32 536870912, i32 268435456}
// CHECK: !20 = !{!"branch_weights", i32 1, i32 715827883, i32 357913942}
