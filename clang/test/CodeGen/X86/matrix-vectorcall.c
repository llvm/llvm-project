// RUN: %clang_cc1 -triple i386-pc-win32 -fenable-matrix -ffreestanding \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HFA,X86
// RUN: %clang_cc1 -triple x86_64-pc-win32 -fenable-matrix -ffreestanding \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,HFA,X64
// RUN: %clang_cc1 -triple i386-pc-win32 -fenable-matrix -ffreestanding \
// RUN:   -fclang-abi-compat=23 -emit-llvm -o - %s | FileCheck %s \
// RUN:   --check-prefixes=CHECK,COMPAT23,X86,X86-COMPAT23
// RUN: %clang_cc1 -triple x86_64-pc-win32 -fenable-matrix -ffreestanding \
// RUN:   -fclang-abi-compat=23 -emit-llvm -o - %s | FileCheck %s \
// RUN:   --check-prefixes=CHECK,COMPAT23,X64,X64-COMPAT23

// x86/x86_64 __vectorcall HVA classification of Clang matrix types. Matrix
// types are flattened like arrays (T NxM -> N*M members of T). vectorcall
// also applies that check to the matrix type itself, so a bare f2x2 is an
// HVA of four floats (passed inreg) after the flattening change.
//
// -fclang-abi-compat=23 does not flatten matrices.

typedef float __attribute__((matrix_type(2, 2))) f2x2;
typedef float __attribute__((matrix_type(3, 3))) f3x3;
typedef double __attribute__((matrix_type(2, 2))) d2x2;
typedef int __attribute__((matrix_type(2, 2))) i2x2;

struct MatrixStruct {
  f2x2 m;
};

struct ArrayStruct {
  float arr[4];
};

struct LargeMatrixStruct {
  f3x3 m;
};

struct DoubleMatrixStruct {
  d2x2 m;
};

struct IntMatrixStruct {
  i2x2 m;
};

struct OverAlignedMatrixStruct {
  f2x2 m;
} __attribute__((aligned(32)));

void __vectorcall take_matrix(f2x2 m) {}
// HFA: define{{.*}} x86_vectorcallcc void @"\01take_matrix@@16"(<4 x float> inreg noundef %{{.*}})
// COMPAT23: define{{.*}} x86_vectorcallcc void @"\01take_matrix@@16"(<4 x float> noundef %{{.*}})

f2x2 __vectorcall return_matrix(void) {
  f2x2 m;
  return m;
}
// CHECK: define{{.*}} x86_vectorcallcc <4 x float> @"\01return_matrix@@0"()

// Four floats, no padding -> HVA passed inreg. Compat 23 uses byval (x86)
// or an indirect pointer (x64).
void __vectorcall take_matrix_struct(struct MatrixStruct s) {}
// HFA: define{{.*}} x86_vectorcallcc void @"\01take_matrix_struct@@16"(%struct.MatrixStruct inreg %{{.*}})
// X86-COMPAT23: define{{.*}} x86_vectorcallcc void @"\01take_matrix_struct@@16"(ptr noundef byval(%struct.MatrixStruct) align 4 %{{.*}})
// X64-COMPAT23: define{{.*}} x86_vectorcallcc void @"\01take_matrix_struct@@16"(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(16) %{{.*}})

struct MatrixStruct __vectorcall return_matrix_struct(void) {
  struct MatrixStruct s;
  return s;
}
// HFA: define{{.*}} x86_vectorcallcc %struct.MatrixStruct @"\01return_matrix_struct@@0"()
// COMPAT23: define{{.*}} x86_vectorcallcc void @"\01return_matrix_struct@@0"(ptr {{.*}}sret(%struct.MatrixStruct)

// float[4] was already an HVA.
void __vectorcall take_array_struct(struct ArrayStruct s) {}
// CHECK: define{{.*}} x86_vectorcallcc void @"\01take_array_struct@@16"(%struct.ArrayStruct inreg %{{.*}})

struct ArrayStruct __vectorcall return_array_struct(void) {
  struct ArrayStruct s;
  return s;
}
// CHECK: define{{.*}} x86_vectorcallcc %struct.ArrayStruct @"\01return_array_struct@@0"()

// Four doubles -> HVA. Compat 23 is byval / indirect.
void __vectorcall take_double_matrix_struct(struct DoubleMatrixStruct s) {}
// HFA: define{{.*}} x86_vectorcallcc void @"\01take_double_matrix_struct@@32"(%struct.DoubleMatrixStruct inreg %{{.*}})
// X86-COMPAT23: define{{.*}} x86_vectorcallcc void @"\01take_double_matrix_struct@@32"(ptr noundef byval(%struct.DoubleMatrixStruct) align 4 %{{.*}})
// X64-COMPAT23: define{{.*}} x86_vectorcallcc void @"\01take_double_matrix_struct@@32"(ptr nofreeobj noundef align 8 dead_on_return dereferenceable(32) %{{.*}})

struct DoubleMatrixStruct __vectorcall return_double_matrix_struct(void) {
  struct DoubleMatrixStruct s;
  return s;
}
// HFA: define{{.*}} x86_vectorcallcc %struct.DoubleMatrixStruct @"\01return_double_matrix_struct@@0"()
// COMPAT23: define{{.*}} x86_vectorcallcc void @"\01return_double_matrix_struct@@0"(ptr {{.*}}sret(%struct.DoubleMatrixStruct)

// Nine floats exceeds the 4-member HVA limit.
void __vectorcall take_large_matrix_struct(struct LargeMatrixStruct s) {}
// X86: define{{.*}} x86_vectorcallcc void @"\01take_large_matrix_struct@@36"(ptr noundef byval(%struct.LargeMatrixStruct) align 4 %{{.*}})
// X64: define{{.*}} x86_vectorcallcc void @"\01take_large_matrix_struct@@40"(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(36) %{{.*}})

// int is not an HVA base type.
void __vectorcall take_int_matrix_struct(struct IntMatrixStruct s) {}
// X86: define{{.*}} x86_vectorcallcc void @"\01take_int_matrix_struct@@16"(ptr noundef byval(%struct.IntMatrixStruct) align 4 %{{.*}})
// X64: define{{.*}} x86_vectorcallcc void @"\01take_int_matrix_struct@@16"(ptr nofreeobj noundef align 4 dead_on_return dereferenceable(16) %{{.*}})

// Tail padding from aligned(32) disqualifies the HVA.
void __vectorcall take_overaligned_matrix_struct(
    struct OverAlignedMatrixStruct s) {}
// X86: define{{.*}} x86_vectorcallcc void @"\01take_overaligned_matrix_struct@@32"(ptr inreg nofreeobj noundef align 32 dead_on_return dereferenceable(32) %{{.*}})
// X64: define{{.*}} x86_vectorcallcc void @"\01take_overaligned_matrix_struct@@32"(ptr nofreeobj noundef align 32 dead_on_return dereferenceable(32) %{{.*}})
