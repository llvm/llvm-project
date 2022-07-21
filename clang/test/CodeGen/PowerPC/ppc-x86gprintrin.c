// REQUIRES: powerpc-registered-target
// RUN: %clang -Xclang -no-opaque-pointers -S -emit-llvm -target powerpc64le-unknown-linux-gnu -mcpu=pwr7 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s
// RUN: %clang -Xclang -no-opaque-pointers -S -emit-llvm -target powerpc64-unknown-linux-gnu -mcpu=pwr7 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s

// RUN: %clang -Xclang -no-opaque-pointers -S -emit-llvm -target powerpc64le-unknown-freebsd13.0 -mcpu=pwr7 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s
// RUN: %clang -Xclang -no-opaque-pointers -S -emit-llvm -target powerpc64-unknown-freebsd13.0 -mcpu=pwr7 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s

// RUN: %clang -Xclang -no-opaque-pointers -S -emit-llvm -target powerpc64-ibm-aix -mcpu=pwr7 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s

#include <x86gprintrin.h>

unsigned short us;
unsigned ui;
unsigned long long ul;

void __attribute__((noinline))
test_bmiintrin() {
  __tzcnt_u16(us);
  __andn_u32(ui, ui);
  _bextr_u32(ui, ui, ui);
  __bextr_u32(ui, ui);
  __blsi_u32(ui);
  _blsi_u32(ui);
  __blsmsk_u32(ui);
  _blsmsk_u32(ui);
  __blsr_u32(ui);
  _blsr_u32(ui);
  __tzcnt_u32(ui);
  _tzcnt_u32(ui);
  __andn_u64(ul, ul);
  _bextr_u64(ul, ui, ui);
  __bextr_u64(ul, ul);
  __blsi_u64(ul);
  _blsi_u64(ul);
  __blsmsk_u64(ul);
  _blsmsk_u64(ul);
  __blsr_u64(ul);
  _blsr_u64(ul);
  __tzcnt_u64(ul);
  _tzcnt_u64(ul);
}

// CHECK-LABEL: @test_bmiintrin

// CHECK-LABEL: define available_externally zeroext i16 @__tzcnt_u16(i16 noundef zeroext %{{[0-9a-zA-Z._]+}})
// CHECK: %[[CONV:[0-9a-zA-Z._]+]] = zext i16 %{{[0-9a-zA-Z._]+}} to i32
// CHECK: %[[CALL:[0-9a-zA-Z._]+]] = call i32 @llvm.cttz.i32(i32 %[[CONV]], i1 false)
// CHECK: trunc i32 %[[CALL]] to i16

// CHECK-LABEL: define available_externally zeroext i32 @__andn_u32(i32 noundef zeroext %{{[0-9a-zA-Z._]+}}, i32 noundef zeroext %{{[0-9a-zA-Z._]+}})
// CHECK: %[[NEG:[0-9a-zA-Z._]+]] = xor i32 %{{[0-9a-zA-Z._]+}}, -1
// CHECK: and i32 %[[NEG]], %1

// CHECK-LABEL: define available_externally zeroext i32 @_bextr_u32(i32 noundef zeroext %{{[0-9a-zA-Z._]+}}, i32 noundef zeroext %{{[0-9a-zA-Z._]+}}, i32 noundef zeroext %{{[0-9a-zA-Z._]+}})
// CHECK: %[[ADD:[0-9a-zA-Z._]+]] = add i32 %{{[0-9a-zA-Z._]+}}, %{{[0-9a-zA-Z._]+}}
// CHECK: %[[SUB:[0-9a-zA-Z._]+]] = sub i32 32, %[[ADD]]
// CHECK: %[[SHL:[0-9a-zA-Z._]+]] = shl i32 %{{[0-9a-zA-Z._]+}}, %[[SUB]]
// CHECK: %[[SUB]]1 = sub i32 32, %{{[0-9a-zA-Z._]+}}
// CHECK: lshr i32 %[[SHL]], %[[SUB]]1

// CHECK-LABEL: define available_externally zeroext i32 @__bextr_u32(i32 noundef zeroext %{{[0-9a-zA-Z._]+}}, i32 noundef zeroext %{{[0-9a-zA-Z._]+}})
// CHECK: %[[AND:[0-9a-zA-Z._]+]] = and i32 %{{[0-9a-zA-Z._]+}}, 255
// CHECK: %[[SHR:[0-9a-zA-Z._]+]] = lshr i32 %{{[0-9a-zA-Z._]+}}, 8
// CHECK: and i32 %[[SHR]], 255
// CHECK: call zeroext i32 @_bextr_u32

// CHECK-LABEL: define available_externally zeroext i32 @__blsi_u32(i32 noundef zeroext %{{[0-9a-zA-Z._]+}})
// CHECK: %[[SUB:[0-9a-zA-Z._]+]] = sub i32 0, %1
// CHECK: and i32 %0, %[[SUB]]

// CHECK-LABEL: define available_externally zeroext i32 @_blsi_u32(i32 noundef zeroext %{{[0-9a-zA-Z._]+}})
// CHECK: call zeroext i32 @__blsi_u32

// CHECK-LABEL: define available_externally zeroext i32 @__blsmsk_u32(i32 noundef zeroext %{{[0-9a-zA-Z._]+}})
// CHECK: %[[SUB:[0-9a-zA-Z._]+]] = sub i32 %{{[0-9a-zA-Z._]+}}, 1
// CHECK: xor i32 %{{[0-9a-zA-Z._]+}}, %[[SUB]]

// CHECK-LABEL: define available_externally zeroext i32 @_blsmsk_u32(i32 noundef zeroext %{{[0-9a-zA-Z._]+}})
// CHECK: call zeroext i32 @__blsmsk_u32

// CHECK-LABEL: define available_externally zeroext i32 @__blsr_u32(i32 noundef zeroext %{{[0-9a-zA-Z._]+}})
// CHECK: %[[SUB:[0-9a-zA-Z._]+]] = sub i32 %{{[0-9a-zA-Z._]+}}, 1
// CHECK: and i32 %{{[0-9a-zA-Z._]+}}, %[[SUB]]

// CHECK-LABEL: define available_externally zeroext i32 @_blsr_u32(i32 noundef zeroext %{{[0-9a-zA-Z._]+}})
// CHECK: call zeroext i32 @__blsr_u32

// CHECK-LABEL: define available_externally zeroext i32 @__tzcnt_u32(i32 noundef zeroext %{{[0-9a-zA-Z._]+}})
// CHECK: call i32 @llvm.cttz.i32(i32 %{{[0-9a-zA-Z._]+}}, i1 false)

// CHECK-LABEL: define available_externally zeroext i32 @_tzcnt_u32(i32 noundef zeroext %{{[0-9a-zA-Z._]+}})
// CHECK: call i32 @llvm.cttz.i32(i32 %{{[0-9a-zA-Z._]+}}, i1 false)

// CHECK-LABEL: define available_externally i64 @__andn_u64(i64 noundef %{{[0-9a-zA-Z._]+}}, i64 noundef %{{[0-9a-zA-Z._]+}})
// CHECK: %[[NEG:[0-9a-zA-Z._]+]] = xor i64 %{{[0-9a-zA-Z._]+}}, -1
// CHECK: and i64 %[[NEG]], %{{[0-9a-zA-Z._]+}}

// CHECK-LABEL: define available_externally i64 @_bextr_u64(i64 noundef %{{[0-9a-zA-Z._]+}}, i32 noundef zeroext %{{[0-9a-zA-Z._]+}}, i32 noundef zeroext %{{[0-9a-zA-Z._]+}})
// CHECK: %[[ADD:[0-9a-zA-Z._]+]] = add i32 %{{[0-9a-zA-Z._]+}}, %{{[0-9a-zA-Z._]+}}
// CHECK: %[[SUB:[0-9a-zA-Z._]+]] = sub i32 64, %[[ADD]]
// CHECK: %[[EXT:[0-9a-zA-Z._]+]] = zext i32 %[[SUB]] to i64
// CHECK: %[[SHL:[0-9a-zA-Z._]+]] = shl i64 %{{[0-9a-zA-Z._]+}}, %[[EXT]]
// CHECK: %[[SUB1:[0-9a-zA-Z._]+]] = sub i32 64, %{{[0-9a-zA-Z._]+}}
// CHECK: %[[EXT2:[0-9a-zA-Z._]+]] = zext i32 %[[SUB1]] to i64
// CHECK: lshr i64 %[[SHL]], %[[EXT2]]

// CHECK-LABEL: define available_externally i64 @__bextr_u64(i64 noundef %{{[0-9a-zA-Z._]+}}, i64 noundef %{{[0-9a-zA-Z._]+}})
// CHECK: %[[AND:[0-9a-zA-Z._]+]] = and i64 %{{[0-9a-zA-Z._]+}}, 255
// CHECK: trunc i64 %[[AND]] to i32
// CHECK: %[[AND1:[0-9a-zA-Z._]+]] = and i64 %{{[0-9a-zA-Z._]+}}, 65280
// CHECK: %[[SHR:[0-9a-zA-Z._]+]] = lshr i64 %[[AND1]], 8
// CHECK: trunc i64 %[[SHR]] to i32
// CHECK: call i64 @_bextr_u64

// CHECK-LABEL: define available_externally i64 @__blsi_u64(i64 noundef %{{[0-9a-zA-Z._]+}})
// CHECK: %[[SUB:[0-9a-zA-Z._]+]] = sub i64 0, %{{[0-9a-zA-Z._]+}}
// CHECK: and i64 %0, %[[SUB]]

// CHECK-LABEL: define available_externally i64 @_blsi_u64(i64 noundef %{{[0-9a-zA-Z._]+}})
// CHECK: call i64 @__blsi_u64

// CHECK-LABEL: define available_externally i64 @__blsmsk_u64(i64 noundef %{{[0-9a-zA-Z._]+}})
// CHECK: %[[SUB:[0-9a-zA-Z._]+]] = sub i64 %{{[0-9a-zA-Z._]+}}, 1
// CHECK: xor i64 %0, %[[SUB]]

// CHECK-LABEL: define available_externally i64 @_blsmsk_u64(i64 noundef %{{[0-9a-zA-Z._]+}})
// CHECK: call i64 @__blsmsk_u64

// CHECK-LABEL: define available_externally i64 @__blsr_u64(i64 noundef %{{[0-9a-zA-Z._]+}})
// CHECK: %[[SUB:[0-9a-zA-Z._]+]] = sub i64 %{{[0-9a-zA-Z._]+}}, 1
// CHECK: and i64 %{{[0-9a-zA-Z._]+}}, %[[SUB]]

// CHECK-LABEL: define available_externally i64 @_blsr_u64(i64 noundef %{{[0-9a-zA-Z._]+}})
// CHECK: call i64 @__blsr_u64

// CHECK-LABEL: define available_externally i64 @__tzcnt_u64(i64 noundef %{{[0-9a-zA-Z._]+}})
// CHECK: %[[CALL:[0-9a-zA-Z._]+]] = call i64 @llvm.cttz.i64(i64 %{{[0-9a-zA-Z._]+}}, i1 false)
// CHECK: %[[CAST:[0-9a-zA-Z._]+]] = trunc i64 %[[CALL]] to i32
// CHECK: sext i32 %[[CAST]] to i64

// CHECK-LABEL: define available_externally i64 @_tzcnt_u64(i64 noundef %{{[0-9a-zA-Z._]+}})
// CHECK: %[[CALL:[0-9a-zA-Z._]+]] = call i64 @llvm.cttz.i64(i64 %{{[0-9a-zA-Z._]+}}, i1 false)
// CHECK: %[[CAST:[0-9a-zA-Z._]+]] = trunc i64 %[[CALL]] to i32
// CHECK: sext i32 %[[CAST]] to i64

void __attribute__((noinline))
test_bmi2intrin() {
  _bzhi_u32(ui, ui);
  _mulx_u32(ui, ui, &ui);
  _bzhi_u64(ul, ul);
  _mulx_u64(ul, ul, &ul);
  _pdep_u64(ul, ul);
  _pext_u64(ul, ul);
  _pdep_u32(ui, ui);
  _pext_u32(ui, ui);
}

// CHECK-LABEL: @test_bmi2intrin

// CHECK-LABEL: define available_externally zeroext i32 @_bzhi_u32(i32 noundef zeroext %{{[0-9a-zA-Z._]+}}, i32 noundef zeroext %{{[0-9a-zA-Z._]+}})
// CHECK: %[[SUB:[0-9a-zA-Z._]+]] = sub i32 32, %{{[0-9a-zA-Z._]+}}
// CHECK: %[[SHL:[0-9a-zA-Z._]+]] = shl i32 %{{[0-9a-zA-Z._]+}}, %[[SUB]]
// CHECK: %[[SUB1:[0-9a-zA-Z._]+]] = sub i32 32, %{{[0-9a-zA-Z._]+}}
// CHECK: lshr i32 %[[SHL]], %[[SUB1]]

// CHECK-LABEL: define available_externally zeroext i32 @_mulx_u32(i32 noundef zeroext %{{[0-9a-zA-Z._]+}}, i32 noundef zeroext %{{[0-9a-zA-Z._]+}}, i32* noundef %{{[0-9a-zA-Z._]+}})
// CHECK: zext i32 %{{[0-9a-zA-Z._]+}} to i64
// CHECK: zext i32 %{{[0-9a-zA-Z._]+}} to i64
// CHECK: %[[SHR:[0-9a-zA-Z._]+]] = lshr i64 %{{[0-9a-zA-Z._]+}}, 32
// CHECK: trunc i64 %[[SHR]] to i32
// CHECK: trunc i64 %{{[0-9a-zA-Z._]+}} to i32

// CHECK-LABEL: define available_externally i64 @_bzhi_u64(i64 noundef %{{[0-9a-zA-Z._]+}}, i64 noundef %{{[0-9a-zA-Z._]+}})
// CHECK: %[[SUB:[0-9a-zA-Z._]+]] = sub i64 64, %{{[0-9a-zA-Z._]+}}
// CHECK: %[[SHL:[0-9a-zA-Z._]+]] = shl i64 %{{[0-9a-zA-Z._]+}}, %[[SUB]]
// CHECK: %[[SUB1:[0-9a-zA-Z._]+]] = sub i64 64, %{{[0-9a-zA-Z._]+}}
// CHECK: lshr i64 %[[SHL]], %[[SUB1]]

// CHECK-LABEL: define available_externally i64 @_mulx_u64(i64 noundef %{{[0-9a-zA-Z._]+}}, i64 noundef %{{[0-9a-zA-Z._]+}}, i64* noundef %{{[0-9a-zA-Z._]+}})
// CHECK: zext i64 %{{[0-9a-zA-Z._]+}} to i128
// CHECK: zext i64 %{{[0-9a-zA-Z._]+}} to i128
// CHECK: %[[SHR:[0-9a-zA-Z._]+]] = lshr i128 %{{[0-9a-zA-Z._]+}}, 64
// CHECK: trunc i128 %[[SHR]] to i64
// CHECK: trunc i128 %{{[0-9a-zA-Z._]+}} to i64

// CHECK-LABEL: define available_externally i64 @_pdep_u64(i64 noundef %{{[0-9a-zA-Z._]+}}, i64 noundef %{{[0-9a-zA-Z._]+}})
// CHECK: %[[CALL:[0-9a-zA-Z._]+]] = call i64 @llvm.ctpop.i64(i64 %{{[0-9a-zA-Z._]+}})
// CHECK: %[[CAST:[0-9a-zA-Z._]+]] = trunc i64 %[[CALL]] to i32
// CHECK: %[[SUB:[0-9a-zA-Z._]+]] = sub nsw i32 64, %[[CAST]]
// CHECK: sext i32 %[[SUB]] to i64
// CHECK: %[[CALL2:[0-9a-zA-Z._]+]] = call i64 @llvm.ctlz.i64(i64 %{{[0-9a-zA-Z._]+}}, i1 false)
// CHECK: %[[CAST2:[0-9a-zA-Z._]+]] = trunc i64 %[[CALL2]] to i32
// CHECK: sext i32 %[[CAST2]] to i64
// CHECK: %[[SUB2:[0-9a-zA-Z._]+]] = sub i64 %{{[0-9a-zA-Z._]+}}, %{{[0-9a-zA-Z._]+}}
// CHECK: shl i64 %{{[0-9a-zA-Z._]+}}, %[[SUB2]]
// CHECK: %[[SHR:[0-9a-zA-Z._]+]] = lshr i64 -9223372036854775808, %{{[0-9a-zA-Z._]+}}
// CHECK: xor i64 %{{[0-9a-zA-Z._]+}}, %[[SHR]]
// CHECK: %[[SHR2:[0-9a-zA-Z._]+]] = lshr i64 -9223372036854775808, %{{[0-9a-zA-Z._]+}}
// CHECK: %[[AND:[0-9a-zA-Z._]+]] = and i64 %{{[0-9a-zA-Z._]+}}, %[[SHR2]]
// CHECK: or i64 %{{[0-9a-zA-Z._]+}}, %[[AND]]
// CHECK: add i64 %{{[0-9a-zA-Z._]+}}, 1

// CHECK-LABEL: define available_externally i64 @_pext_u64(i64 noundef %{{[0-9a-zA-Z._]+}}, i64 noundef %{{[0-9a-zA-Z._]+}})
// CHECK: %[[CALL:[0-9a-zA-Z._]+]] = call i1 @llvm.is.constant.i64(i64 %{{[0-9a-zA-Z._]+}})
// CHECK: br i1 %[[CALL]], label %[[TRUECOND:[0-9a-zA-Z._]+]], label %[[FALSECOND:[0-9a-zA-Z._]+]]
// CHECK: [[TRUECOND]]:
// CHECK: %[[CALL2:[0-9a-zA-Z._]+]] = call i64 @llvm.ctpop.i64(i64 %{{[0-9a-zA-Z._]+}})
// CHECK: call i64 @llvm.ctlz.i64(i64 %{{[0-9a-zA-Z._]+}}, i1 false)
// CHECK: %[[SHL:[0-9a-zA-Z._]+]] = shl i64 %{{[0-9a-zA-Z._]+}}, 8
// CHECK: or i64 %[[SHL]], %{{[0-9a-zA-Z._]+}}
// CHECK: %[[SHR:[0-9a-zA-Z._]+]] = lshr i64 -9223372036854775808, %{{[0-9a-zA-Z._]+}}
// CHECK: xor i64 %{{[0-9a-zA-Z._]+}}, %[[SHR]]
// CHECK: add nsw i64 %{{[0-9a-zA-Z._]+}}, 1
// CHECK: call i64 @llvm.ppc.bpermd
// CHECK: [[FALSECOND]]:
// CHECK: call i64 @llvm.ctpop.i64(i64 %{{[0-9a-zA-Z._]+}})
// CHECK: call i64 @llvm.ctlz.i64(i64 %{{[0-9a-zA-Z._]+}}, i1 false)
// CHECK: %[[SHR2:[0-9a-zA-Z._]+]] = lshr i64 -9223372036854775808, %{{[0-9a-zA-Z._]+}}
// CHECK: %[[AND:[0-9a-zA-Z._]+]] = and i64 %{{[0-9a-zA-Z._]+}}, %[[SHR2]]
// CHECK: %[[SUB:[0-9a-zA-Z._]+]] = sub i64 %{{[0-9a-zA-Z._]+}}, %{{[0-9a-zA-Z._]+}}
// CHECK: lshr i64 %[[AND]], %[[SUB]]
// CHECK: %[[SHR3:[0-9a-zA-Z._]+]] = lshr i64 -9223372036854775808, %{{[0-9a-zA-Z._]+}}
// CHECK: xor i64 %{{[0-9a-zA-Z._]+}}, %[[SHR3]]
// CHECK: or i64 %{{[0-9a-zA-Z._]+}}, %{{[0-9a-zA-Z._]+}}
// CHECK: add i64 %{{[0-9a-zA-Z._]+}}, 1

// CHECK-LABEL: define available_externally zeroext i32 @_pdep_u32(i32 noundef zeroext %{{[0-9a-zA-Z._]+}}, i32 noundef zeroext %{{[0-9a-zA-Z._]+}})
// CHECK: %[[CONV:[0-9a-zA-Z._]+]] = zext i32 %{{[0-9a-zA-Z._]+}} to i64
// CHECK: %[[CONV1:[0-9a-zA-Z._]+]] = zext i32 %{{[0-9a-zA-Z._]+}} to i64
// CHECK: %[[CALL:[0-9a-zA-Z._]+]] = call i64 @_pdep_u64(i64 noundef %[[CONV]], i64 noundef %[[CONV1]])
// CHECK: trunc i64 %[[CALL]] to i32

// CHECK-LABEL: define available_externally zeroext i32 @_pext_u32(i32 noundef zeroext %{{[0-9a-zA-Z._]+}}, i32 noundef zeroext %{{[0-9a-zA-Z._]+}})
// CHECK: %[[CONV:[0-9a-zA-Z._]+]] = zext i32 %{{[0-9a-zA-Z._]+}} to i64
// CHECK: %[[CONV1:[0-9a-zA-Z._]+]] = zext i32 %{{[0-9a-zA-Z._]+}} to i64
// CHECK: %[[CALL:[0-9a-zA-Z._]+]] = call i64 @_pext_u64(i64 noundef %[[CONV]], i64 noundef %[[CONV1]])
// CHECK: trunc i64 %[[CALL]] to i32
