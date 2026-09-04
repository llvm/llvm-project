// TODO(cir): drop -fno-clangir-call-conv-lowering once CallConvLowering
// supports _BitInt wider than 128 bits.
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-call-conv-lowering -emit-llvm -DGLOBAL %s -o - 2>&1 | FileCheck %s --check-prefix=GLOBAL
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-call-conv-lowering -emit-llvm -DALLOCA %s -o - 2>&1 | FileCheck %s --check-prefix=ALLOCA
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-call-conv-lowering -emit-llvm -DSTORE %s -o - 2>&1 | FileCheck %s --check-prefix=STORE
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-call-conv-lowering -emit-llvm -DLOAD %s -o - 2>&1 | FileCheck %s --check-prefix=LOAD
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-call-conv-lowering -emit-llvm -DSTRUCT %s -o - 2>&1 | FileCheck %s --check-prefix=STRUCT
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-call-conv-lowering -emit-llvm -DARRAY %s -o - 2>&1 | FileCheck %s --check-prefix=ARRAY

#ifdef GLOBAL
signed _BitInt(129) g129 = 1;
// GLOBAL: NYI: lowering global of a type with no memory representation
#endif

#ifdef ALLOCA
signed _BitInt(129) use_local(signed _BitInt(129) a) {
  signed _BitInt(129) x = a;
  return x;
}
// ALLOCA: NYI: lowering alloca of a type with no memory representation
#endif

#ifdef STORE
void store_lit(signed _BitInt(129) *p) { *p = (signed _BitInt(129))1; }
// STORE: NYI: lowering store of a type with no memory representation
#endif

#ifdef LOAD
int load_cmp(signed _BitInt(129) *p) { return *p != 0; }
// LOAD: NYI: lowering load of a type with no memory representation
#endif

#ifdef STRUCT
// FIXME: Make sure we test that the layout of this and the array struct are
// 'correct' when this lowering is completed.
struct HasWide129 {
  int i;
  signed _BitInt(129) bi;
};
struct HasWide129 g_struct;
// STRUCT: NYI: lowering global of a type with no memory representation
#endif

#ifdef ARRAY
struct HasWide129Array {
  int i;
  signed _BitInt(129) bi[2];
};
struct HasWide129Array g_array;
// ARRAY: NYI: lowering global of a type with no memory representation
#endif
