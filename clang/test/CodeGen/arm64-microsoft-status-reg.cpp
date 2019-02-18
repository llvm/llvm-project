// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple arm64-windows -fms-compatibility -emit-llvm -S \
// RUN: -o - %s | FileCheck %s -check-prefix CHECK-ASM

// RUN: %clang_cc1 -triple arm64-windows -fms-compatibility -emit-llvm \
// RUN: -o - %s | FileCheck %s -check-prefix CHECK-IR

// From winnt.h
#define ARM64_SYSREG(op0, op1, crn, crm, op2) \
        ( ((op0 & 1) << 14) | \
          ((op1 & 7) << 11) | \
          ((crn & 15) << 7) | \
          ((crm & 15) << 3) | \
          ((op2 & 7) << 0) )

#define ARM64_CNTVCT            ARM64_SYSREG(3,3,14, 0,2)  // Generic Timer counter register
#define ARM64_PMCCNTR_EL0       ARM64_SYSREG(3,3, 9,13,0)  // Cycle Count Register [CP15_PMCCNTR]
#define ARM64_PMSELR_EL0        ARM64_SYSREG(3,3, 9,12,5)  // Event Counter Selection Register [CP15_PMSELR]
#define ARM64_PMXEVCNTR_EL0     ARM64_SYSREG(3,3, 9,13,2)  // Event Count Register [CP15_PMXEVCNTR]
#define ARM64_PMXEVCNTRn_EL0(n) ARM64_SYSREG(3,3,14, 8+((n)/8), (n)%8)    // Direct Event Count Register [n/a]
#define ARM64_TPIDR_EL0         ARM64_SYSREG(3,3,13, 0,2)  // Thread ID Register, User Read/Write [CP15_TPIDRURW]
#define ARM64_TPIDRRO_EL0       ARM64_SYSREG(3,3,13, 0,3)  // Thread ID Register, User Read Only [CP15_TPIDRURO]
#define ARM64_TPIDR_EL1         ARM64_SYSREG(3,0,13, 0,4)  // Thread ID Register, Privileged Only [CP15_TPIDRPRW]

void check_ReadWriteStatusReg(int v) {
  int ret;
  ret = _ReadStatusReg(ARM64_CNTVCT);
// CHECK-ASM: mrs     x8, CNTVCT_EL0
// CHECK-IR: call i64 @llvm.read_register.i64(metadata ![[MD2:.*]])

  ret = _ReadStatusReg(ARM64_PMCCNTR_EL0);
// CHECK-ASM: mrs     x8, PMCCNTR_EL0
// CHECK-IR: call i64 @llvm.read_register.i64(metadata ![[MD3:.*]])

  ret = _ReadStatusReg(ARM64_PMSELR_EL0);
// CHECK-ASM: mrs     x8, PMSELR_EL0
// CHECK-IR: call i64 @llvm.read_register.i64(metadata ![[MD4:.*]])

  ret = _ReadStatusReg(ARM64_PMXEVCNTR_EL0);
// CHECK-ASM: mrs     x8, PMXEVCNTR_EL0
// CHECK-IR: call i64 @llvm.read_register.i64(metadata ![[MD5:.*]])

  ret = _ReadStatusReg(ARM64_PMXEVCNTRn_EL0(0));
// CHECK-ASM: mrs     x8, PMEVCNTR0_EL0
// CHECK-IR: call i64 @llvm.read_register.i64(metadata ![[MD6:.*]])

  ret = _ReadStatusReg(ARM64_PMXEVCNTRn_EL0(1));
// CHECK-ASM: mrs     x8, PMEVCNTR1_EL0
// CHECK-IR: call i64 @llvm.read_register.i64(metadata ![[MD7:.*]])

  ret = _ReadStatusReg(ARM64_PMXEVCNTRn_EL0(30));
// CHECK-ASM: mrs     x8, PMEVCNTR30_EL0
// CHECK-IR: call i64 @llvm.read_register.i64(metadata ![[MD8:.*]])

  ret = _ReadStatusReg(ARM64_TPIDR_EL0);
// CHECK-ASM: mrs     x8, TPIDR_EL0
// CHECK-IR: call i64 @llvm.read_register.i64(metadata ![[MD9:.*]])

  ret = _ReadStatusReg(ARM64_TPIDRRO_EL0);
// CHECK-ASM: mrs     x8, TPIDRRO_EL0
// CHECK-IR: call i64 @llvm.read_register.i64(metadata ![[MD10:.*]])

  ret = _ReadStatusReg(ARM64_TPIDR_EL1);
// CHECK-ASM: mrs     x8, TPIDR_EL1
// CHECK-IR: call i64 @llvm.read_register.i64(metadata ![[MD11:.*]])


  _WriteStatusReg(ARM64_CNTVCT, v);
// CHECK-ASM: msr     S3_3_C14_C0_2, x8
// CHECK-IR: call void @llvm.write_register.i64(metadata ![[MD2:.*]], i64 {{%.*}})

  _WriteStatusReg(ARM64_PMCCNTR_EL0, v);
// CHECK-ASM: msr     PMCCNTR_EL0, x8
// CHECK-IR: call void @llvm.write_register.i64(metadata ![[MD3:.*]], i64 {{%.*}})

  _WriteStatusReg(ARM64_PMSELR_EL0, v);
// CHECK-ASM: msr     PMSELR_EL0, x8
// CHECK-IR: call void @llvm.write_register.i64(metadata ![[MD4:.*]], i64 {{%.*}})

  _WriteStatusReg(ARM64_PMXEVCNTR_EL0, v);
// CHECK-ASM: msr     PMXEVCNTR_EL0, x8
// CHECK-IR: call void @llvm.write_register.i64(metadata ![[MD5:.*]], i64 {{%.*}})

  _WriteStatusReg(ARM64_PMXEVCNTRn_EL0(0), v);
// CHECK-ASM: msr     PMEVCNTR0_EL0, x8
// CHECK-IR: call void @llvm.write_register.i64(metadata ![[MD6:.*]], i64 {{%.*}})

  _WriteStatusReg(ARM64_PMXEVCNTRn_EL0(1), v);
// CHECK-ASM: msr     PMEVCNTR1_EL0, x8
// CHECK-IR: call void @llvm.write_register.i64(metadata ![[MD7:.*]], i64 {{%.*}})

  _WriteStatusReg(ARM64_PMXEVCNTRn_EL0(30), v);
// CHECK-ASM: msr     PMEVCNTR30_EL0, x8
// CHECK-IR: call void @llvm.write_register.i64(metadata ![[MD8:.*]], i64 {{%.*}})

  _WriteStatusReg(ARM64_TPIDR_EL0, v);
// CHECK-ASM: msr     TPIDR_EL0, x8
// CHECK-IR: call void @llvm.write_register.i64(metadata ![[MD9:.*]], i64 {{%.*}})

  _WriteStatusReg(ARM64_TPIDRRO_EL0, v);
// CHECK-ASM: msr     TPIDRRO_EL0, x8
// CHECK-IR: call void @llvm.write_register.i64(metadata ![[MD10:.*]], i64 {{%.*}})

  _WriteStatusReg(ARM64_TPIDR_EL1, v);
// CHECK-ASM: msr     TPIDR_EL1, x8
// CHECK-IR: call void @llvm.write_register.i64(metadata ![[MD11:.*]], i64 {{%.*}})
}

// CHECK-IR: ![[MD2]] = !{!"3:3:14:0:2"}
// CHECK-IR: ![[MD3]] = !{!"3:3:9:13:0"}
// CHECK-IR: ![[MD4]] = !{!"3:3:9:12:5"}
// CHECK-IR: ![[MD5]] = !{!"3:3:9:13:2"}
// CHECK-IR: ![[MD6]] = !{!"3:3:14:8:0"}
// CHECK-IR: ![[MD7]] = !{!"3:3:14:8:1"}
// CHECK-IR: ![[MD8]] = !{!"3:3:14:11:6"}
// CHECK-IR: ![[MD9]] = !{!"3:3:13:0:2"}
// CHECK-IR: ![[MD10]] = !{!"3:3:13:0:3"}
// CHECK-IR: ![[MD11]] = !{!"3:0:13:0:4"}
