// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -DGLOBAL %s -o - 2>&1 | FileCheck %s --check-prefix=GLOBAL
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -DALLOCA %s -o - 2>&1 | FileCheck %s --check-prefix=ALLOCA
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -DSTORE %s -o - 2>&1 | FileCheck %s --check-prefix=STORE
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -DLOAD %s -o - 2>&1 | FileCheck %s --check-prefix=LOAD

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
