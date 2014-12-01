// RUN: %clang_cc1 -triple arm-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=ARM
// RUN: %clang_cc1 -triple powerpc-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=PPC32
// RUN: %clang_cc1 -triple powerpc64-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=PPC64
// RUN: %clang_cc1 -triple mipsel-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=MIPS32
// RUN: %clang_cc1 -triple mips64el-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefix=MIPS64

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

// ARM-LABEL: define arm_aapcscc void @test1
// ARM: = call arm_aapcscc zeroext i8 @__atomic_load_1(i8* @c1
// ARM: call arm_aapcscc void @__atomic_store_1(i8* @c1, i8 zeroext
// ARM: = call arm_aapcscc zeroext i16 @__atomic_load_2(i8* bitcast (i16* @s1 to i8*)
// ARM: call arm_aapcscc void @__atomic_store_2(i8* bitcast (i16* @s1 to i8*), i16 zeroext
// ARM: = call arm_aapcscc i32 @__atomic_load_4(i8* bitcast (i32* @i1 to i8*)
// ARM: call arm_aapcscc void @__atomic_store_4(i8* bitcast (i32* @i1 to i8*), i32
// ARM: = call arm_aapcscc i64 @__atomic_load_8(i8* bitcast (i64* @ll1 to i8*)
// ARM: call arm_aapcscc void @__atomic_store_8(i8* bitcast (i64* @ll1 to i8*), i64
// ARM: call arm_aapcscc void @__atomic_load(i32 100, i8* getelementptr inbounds ([100 x i8]* @a1, i32 0, i32 0), i8* getelementptr inbounds ([100 x i8]* @a2, i32 0, i32 0)
// ARM: call arm_aapcscc void @__atomic_store(i32 100, i8* getelementptr inbounds ([100 x i8]* @a1, i32 0, i32 0), i8* getelementptr inbounds ([100 x i8]* @a2, i32 0, i32 0)

// PPC32-LABEL: define void @test1
// PPC32: = load atomic i8* @c1 seq_cst
// PPC32: store atomic i8 {{.*}}, i8* @c1 seq_cst
// PPC32: = load atomic i16* @s1 seq_cst
// PPC32: store atomic i16 {{.*}}, i16* @s1 seq_cst
// PPC32: = load atomic i32* @i1 seq_cst
// PPC32: store atomic i32 {{.*}}, i32* @i1 seq_cst
// PPC32: = call i64 @__atomic_load_8(i8* bitcast (i64* @ll1 to i8*)
// PPC32: call void @__atomic_store_8(i8* bitcast (i64* @ll1 to i8*), i64
// PPC32: call void @__atomic_load(i32 100, i8* getelementptr inbounds ([100 x i8]* @a1, i32 0, i32 0), i8* getelementptr inbounds ([100 x i8]* @a2, i32 0, i32 0)
// PPC32: call void @__atomic_store(i32 100, i8* getelementptr inbounds ([100 x i8]* @a1, i32 0, i32 0), i8* getelementptr inbounds ([100 x i8]* @a2, i32 0, i32 0)

// PPC64-LABEL: define void @test1
// PPC64: = load atomic i8* @c1 seq_cst
// PPC64: store atomic i8 {{.*}}, i8* @c1 seq_cst
// PPC64: = load atomic i16* @s1 seq_cst
// PPC64: store atomic i16 {{.*}}, i16* @s1 seq_cst
// PPC64: = load atomic i32* @i1 seq_cst
// PPC64: store atomic i32 {{.*}}, i32* @i1 seq_cst
// PPC64: = load atomic i64* @ll1 seq_cst
// PPC64: store atomic i64 {{.*}}, i64* @ll1 seq_cst
// PPC64: call void @__atomic_load(i64 100, i8* getelementptr inbounds ([100 x i8]* @a1, i32 0, i32 0), i8* getelementptr inbounds ([100 x i8]* @a2, i32 0, i32 0)
// PPC64: call void @__atomic_store(i64 100, i8* getelementptr inbounds ([100 x i8]* @a1, i32 0, i32 0), i8* getelementptr inbounds ([100 x i8]* @a2, i32 0, i32 0)

// MIPS32-LABEL: define void @test1
// MIPS32: = load atomic i8* @c1 seq_cst
// MIPS32: store atomic i8 {{.*}}, i8* @c1 seq_cst
// MIPS32: = load atomic i16* @s1 seq_cst
// MIPS32: store atomic i16 {{.*}}, i16* @s1 seq_cst
// MIPS32: = load atomic i32* @i1 seq_cst
// MIPS32: store atomic i32 {{.*}}, i32* @i1 seq_cst
// MIPS32: call i64 @__atomic_load_8(i8* bitcast (i64* @ll1 to i8*)
// MIPS32: call void @__atomic_store_8(i8* bitcast (i64* @ll1 to i8*), i64
// MIPS32: call void @__atomic_load(i32 zeroext 100, i8* getelementptr inbounds ([100 x i8]* @a1, i32 0, i32 0), i8* getelementptr inbounds ([100 x i8]* @a2, i32 0, i32 0)
// MIPS32: call void @__atomic_store(i32 zeroext 100, i8* getelementptr inbounds ([100 x i8]* @a1, i32 0, i32 0), i8* getelementptr inbounds ([100 x i8]* @a2, i32 0, i32 0)

// MIPS64-LABEL: define void @test1
// MIPS64: = load atomic i8* @c1 seq_cst
// MIPS64: store atomic i8 {{.*}}, i8* @c1 seq_cst
// MIPS64: = load atomic i16* @s1 seq_cst
// MIPS64: store atomic i16 {{.*}}, i16* @s1 seq_cst
// MIPS64: = load atomic i32* @i1 seq_cst
// MIPS64: store atomic i32 {{.*}}, i32* @i1 seq_cst
// MIPS64: = load atomic i64* @ll1 seq_cst
// MIPS64: store atomic i64 {{.*}}, i64* @ll1 seq_cst
// MIPS64: call void @__atomic_load(i64 zeroext 100, i8* getelementptr inbounds ([100 x i8]* @a1, i32 0, i32 0)
// MIPS64: call void @__atomic_store(i64 zeroext 100, i8* getelementptr inbounds ([100 x i8]* @a1, i32 0, i32 0), i8* getelementptr inbounds ([100 x i8]* @a2, i32 0, i32 0)
}
