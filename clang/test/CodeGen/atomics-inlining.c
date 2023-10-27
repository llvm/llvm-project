// RUN: %clang_cc1 -triple arm-linux-gnueabi -emit-llvm %s -o - | FileCheck %s -check-prefix=ARM
// RUN: %clang_cc1 -triple powerpc-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=PPC32
// RUN: %clang_cc1 -triple powerpc64-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=PPC64
// RUN: %clang_cc1 -triple mipsel-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=MIPS32
// RUN: %clang_cc1 -triple mipsisa32r6el-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=MIPS32
// RUN: %clang_cc1 -triple mips64el-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=MIPS64
// RUN: %clang_cc1 -triple mips64el-linux-gnuabi64 -emit-llvm %s -o - | FileCheck %s -check-prefix=MIPS64
// RUN: %clang_cc1 -triple mipsisa64r6el-linux-gnuabi64 -emit-llvm %s -o - | FileCheck %s -check-prefix=MIPS64
// RUN: %clang_cc1 -triple sparc-unknown-eabi -emit-llvm %s -o - | FileCheck %s -check-prefix=SPARCV8 -check-prefix=SPARC
// RUN: %clang_cc1 -triple sparcv9-unknown-eabi -emit-llvm %s -o - | FileCheck %s -check-prefix=SPARCV9 -check-prefix=SPARC
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -emit-llvm %s -o - | FileCheck %s -check-prefix=NVPTX

unsigned char c1, c2;
unsigned short s1, s2;
unsigned int i1, i2;
unsigned long long ll1, ll2;
unsigned char a1[100], a2[100];

enum memory_order {
  memory_order_relaxed,
  memory_order_consume,
  memory_order_acquire,
  memory_order_release,
  memory_order_acq_rel,
  memory_order_seq_cst
};

void test1(void) {
  (void)__atomic_load(&c1, &c2, memory_order_seq_cst);
  (void)__atomic_store(&c1, &c2, memory_order_seq_cst);
  (void)__atomic_load(&s1, &s2, memory_order_seq_cst);
  (void)__atomic_store(&s1, &s2, memory_order_seq_cst);
  (void)__atomic_load(&i1, &i2, memory_order_seq_cst);
  (void)__atomic_store(&i1, &i2, memory_order_seq_cst);
  (void)__atomic_load(&ll1, &ll2, memory_order_seq_cst);
  (void)__atomic_store(&ll1, &ll2, memory_order_seq_cst);
  (void)__atomic_load(&a1, &a2, memory_order_seq_cst);
  (void)__atomic_store(&a1, &a2, memory_order_seq_cst);

// ARM-LABEL: define{{.*}} void @test1
// ARM: = call{{.*}} zeroext i8 @__atomic_load_1(ptr noundef @c1
// ARM: call{{.*}} void @__atomic_store_1(ptr noundef @c1, i8 noundef zeroext
// ARM: = call{{.*}} zeroext i16 @__atomic_load_2(ptr noundef @s1
// ARM: call{{.*}} void @__atomic_store_2(ptr noundef @s1, i16 noundef zeroext
// ARM: = call{{.*}} i32 @__atomic_load_4(ptr noundef @i1
// ARM: call{{.*}} void @__atomic_store_4(ptr noundef @i1, i32 noundef
// ARM: = call{{.*}} i64 @__atomic_load_8(ptr noundef @ll1
// ARM: call{{.*}} void @__atomic_store_8(ptr noundef @ll1, i64 noundef
// ARM: call{{.*}} void @__atomic_load(i32 noundef 100, ptr noundef @a1, ptr noundef @a2
// ARM: call{{.*}} void @__atomic_store(i32 noundef 100, ptr noundef @a1, ptr noundef @a2

// PPC32-LABEL: define{{.*}} void @test1
// PPC32: = load atomic i8, ptr @c1 seq_cst, align 1
// PPC32: store atomic i8 {{.*}}, ptr @c1 seq_cst, align 1
// PPC32: = load atomic i16, ptr @s1 seq_cst, align 2
// PPC32: store atomic i16 {{.*}}, ptr @s1 seq_cst, align 2
// PPC32: = load atomic i32, ptr @i1 seq_cst, align 4
// PPC32: store atomic i32 {{.*}}, ptr @i1 seq_cst, align 4
// PPC32: = call i64 @__atomic_load_8(ptr noundef @ll1
// PPC32: call void @__atomic_store_8(ptr noundef @ll1, i64
// PPC32: call void @__atomic_load(i32 noundef 100, ptr noundef @a1, ptr noundef @a2
// PPC32: call void @__atomic_store(i32 noundef 100, ptr noundef @a1, ptr noundef @a2

// PPC64-LABEL: define{{.*}} void @test1
// PPC64: = load atomic i8, ptr @c1 seq_cst, align 1
// PPC64: store atomic i8 {{.*}}, ptr @c1 seq_cst, align 1
// PPC64: = load atomic i16, ptr @s1 seq_cst, align 2
// PPC64: store atomic i16 {{.*}}, ptr @s1 seq_cst, align 2
// PPC64: = load atomic i32, ptr @i1 seq_cst, align 4
// PPC64: store atomic i32 {{.*}}, ptr @i1 seq_cst, align 4
// PPC64: = load atomic i64, ptr @ll1 seq_cst, align 8
// PPC64: store atomic i64 {{.*}}, ptr @ll1 seq_cst, align 8
// PPC64: call void @__atomic_load(i64 noundef 100, ptr noundef @a1, ptr noundef @a2
// PPC64: call void @__atomic_store(i64 noundef 100, ptr noundef @a1, ptr noundef @a2

// MIPS32-LABEL: define{{.*}} void @test1
// MIPS32: = load atomic i8, ptr @c1 seq_cst, align 1
// MIPS32: store atomic i8 {{.*}}, ptr @c1 seq_cst, align 1
// MIPS32: = load atomic i16, ptr @s1 seq_cst, align 2
// MIPS32: store atomic i16 {{.*}}, ptr @s1 seq_cst, align 2
// MIPS32: = load atomic i32, ptr @i1 seq_cst, align 4
// MIPS32: store atomic i32 {{.*}}, ptr @i1 seq_cst, align 4
// MIPS32: call i64 @__atomic_load_8(ptr noundef @ll1
// MIPS32: call void @__atomic_store_8(ptr noundef @ll1, i64
// MIPS32: call void @__atomic_load(i32 noundef signext 100, ptr noundef @a1, ptr noundef @a2
// MIPS32: call void @__atomic_store(i32 noundef signext 100, ptr noundef @a1, ptr noundef @a2

// MIPS64-LABEL: define{{.*}} void @test1
// MIPS64: = load atomic i8, ptr @c1 seq_cst, align 1
// MIPS64: store atomic i8 {{.*}}, ptr @c1 seq_cst, align 1
// MIPS64: = load atomic i16, ptr @s1 seq_cst, align 2
// MIPS64: store atomic i16 {{.*}}, ptr @s1 seq_cst, align 2
// MIPS64: = load atomic i32, ptr @i1 seq_cst, align 4
// MIPS64: store atomic i32 {{.*}}, ptr @i1 seq_cst, align 4
// MIPS64: = load atomic i64, ptr @ll1 seq_cst, align 8
// MIPS64: store atomic i64 {{.*}}, ptr @ll1 seq_cst, align 8
// MIPS64: call void @__atomic_load(i64 noundef zeroext 100, ptr noundef @a1
// MIPS64: call void @__atomic_store(i64 noundef zeroext 100, ptr noundef @a1, ptr noundef @a2

// SPARC-LABEL: define{{.*}} void @test1
// SPARC: = load atomic i8, ptr @c1 seq_cst, align 1
// SPARC: store atomic i8 {{.*}}, ptr @c1 seq_cst, align 1
// SPARC: = load atomic i16, ptr @s1 seq_cst, align 2
// SPARC: store atomic i16 {{.*}}, ptr @s1 seq_cst, align 2
// SPARC: = load atomic i32, ptr @i1 seq_cst, align 4
// SPARC: store atomic i32 {{.*}}, ptr @i1 seq_cst, align 4
// SPARCV8: call i64 @__atomic_load_8(ptr noundef @ll1
// SPARCV8: call void @__atomic_store_8(ptr noundef @ll1, i64
// SPARCV9: load atomic i64, ptr @ll1 seq_cst, align 8
// SPARCV9: store atomic i64 {{.*}}, ptr @ll1 seq_cst, align 8
// SPARCV8: call void @__atomic_load(i32 noundef 100, ptr noundef @a1, ptr noundef @a2
// SPARCV8: call void @__atomic_store(i32 noundef 100, ptr noundef @a1, ptr noundef @a2

// NVPTX-LABEL: define{{.*}} void @test1
// NVPTX: = load atomic i8, ptr @c1 seq_cst, align 1
// NVPTX: store atomic i8 {{.*}}, ptr @c1 seq_cst, align 1
// NVPTX: = load atomic i16, ptr @s1 seq_cst, align 2
// NVPTX: store atomic i16 {{.*}}, ptr @s1 seq_cst, align 2
// NVPTX: = load atomic i32, ptr @i1 seq_cst, align 4
// NVPTX: store atomic i32 {{.*}}, ptr @i1 seq_cst, align 4
// NVPTX: = load atomic i64, ptr @ll1 seq_cst, align 8
// NVPTX: store atomic i64 {{.*}}, ptr @ll1 seq_cst, align 8
// NVPTX: call void @__atomic_load(i64 noundef 100, ptr noundef @a1, ptr noundef @a2, i32 noundef 5)
// NVPTX: call void @__atomic_store(i64 noundef 100, ptr noundef @a1, ptr noundef @a2, i32 noundef 5)

}
