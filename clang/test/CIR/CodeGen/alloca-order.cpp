// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefixes=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefixes=LLVM,OGCG --input-file=%t.ll %s

void use(int *);

int main(int argc, char **argv) {
  {
    int x = 1;
    use(&x);
  }
  return 0;
}

// CIR:      cir.func {{.*}} @main(
// CIR-NEXT:   %[[RETVAL:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT:   %[[ARGC:.*]] = cir.alloca "argc" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT:   %[[ARGV:.*]] = cir.alloca "argv" align(8) init : !cir.ptr<!cir.ptr<!cir.ptr<!s8i>>>
// CIR-NEXT:   %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:   cir.store %[[ZERO]], %[[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   cir.store %{{.+}}, %[[ARGC]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   cir.store %{{.+}}, %[[ARGV]] : !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s8i>>>
// CIR-NEXT:   cir.scope {
// CIR-NEXT:     %[[X:.*]] = cir.alloca "x" align(4) init : !cir.ptr<!s32i>

// LLVM:      define dso_local noundef i32 @main(i32 noundef %{{.+}}, ptr noundef %{{.+}})
// LLVM:        %[[RETVAL:.*]] = alloca i32, align 4
// LLVM-NEXT:   %[[ARGC:.*]] = alloca i32, align 4
// LLVM-NEXT:   %[[ARGV:.*]] = alloca ptr, align 8
// LLVM-NEXT:   %[[X:.*]] = alloca i32, align 4
// LLVM-NEXT:   store i32 0, ptr %[[RETVAL]], align 4
// LLVM-NEXT:   store i32 %{{.+}}, ptr %[[ARGC]], align 4
// LLVM-NEXT:   store ptr %{{.+}}, ptr %[[ARGV]], align 8
// LLVM:        store i32 1, ptr %[[X]], align 4

int pick(int a, int b) {
  if (a)
    return a;
  {
    int t = b;
    use(&t);
  }
  return b;
}

// CIR:      cir.func {{.*}} @_Z4pickii(
// CIR-NEXT:   %[[RETVAL:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT:   %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT:   %[[B:.*]] = cir.alloca "b" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT:   cir.store %{{.+}}, %[[A]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:   cir.store %{{.+}}, %[[B]] : !s32i, !cir.ptr<!s32i>

// LLVM:      define dso_local noundef i32 @_Z4pickii(i32 noundef %{{.+}}, i32 noundef %{{.+}})
// LLVM:        %[[RETVAL:.*]] = alloca i32, align 4
// LLVM-NEXT:   %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT:   %[[B:.*]] = alloca i32, align 4
// LLVM-NEXT:   %[[T:.*]] = alloca i32, align 4
// LLVM-NEXT:   store i32 %{{.+}}, ptr %[[A]], align 4
// LLVM-NEXT:   store i32 %{{.+}}, ptr %[[B]], align 4
// LLVM:        store i32 %{{.+}}, ptr %[[T]], align 4
// LLVM:        store i32 %{{.+}}, ptr %[[RETVAL]], align 4

int late(int a) {
  int x = a;
  use(&x);
  {
    int y = x;
    use(&y);
  }
  int z = a;
  use(&z);
  if (z)
    return x;
  return z;
}

// CIR:      cir.func {{.*}} @_Z4latei(
// CIR-NEXT:   %[[RETVAL:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT:   %[[A:.*]] = cir.alloca "a" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT:   %[[X:.*]] = cir.alloca "x" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT:   %[[Z:.*]] = cir.alloca "z" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT:   cir.store %{{.+}}, %[[A]] : !s32i, !cir.ptr<!s32i>
// CIR:        cir.scope {
// CIR:          %[[Y:.*]] = cir.alloca "y" align(4) init : !cir.ptr<!s32i>

// LLVM:      define dso_local noundef i32 @_Z4latei(i32 noundef %{{.+}})
// LLVM:          %[[RETVAL:.*]] = alloca i32, align 4
// LLVM-NEXT:     %[[A:.*]] = alloca i32, align 4
// LLVM-NEXT:     %[[X:.*]] = alloca i32, align 4
// CIR places z, a function-scope local, ahead of the nested block's y.
// LLVMCIR-NEXT:  %[[Z:.*]] = alloca i32, align 4
// LLVMCIR-NEXT:  %[[Y:.*]] = alloca i32, align 4
// OGCG-NEXT:     %[[Y:.*]] = alloca i32, align 4
// OGCG-NEXT:     %[[Z:.*]] = alloca i32, align 4
// LLVM-NEXT:     store i32 %{{.+}}, ptr %[[A]], align 4
// LLVM:          store i32 %{{.+}}, ptr %[[X]], align 4
// LLVM:          store i32 %{{.+}}, ptr %[[Y]], align 4
// LLVM:          store i32 %{{.+}}, ptr %[[Z]], align 4
// LLVM:          store i32 %{{.+}}, ptr %[[RETVAL]], align 4

struct S {
  int m;
  void set(int k);
  int pick(int k);
};

void S::set(int k) {
  {
    int y = k;
    use(&y);
  }
  m = k;
}

// CIR:      cir.func {{.*}} @_ZN1S3setEi(
// CIR-NEXT:   %[[THIS:.*]] = cir.alloca "this" align(8) init : !cir.ptr<!cir.ptr<!rec_S>>
// CIR-NEXT:   %[[K:.*]] = cir.alloca "k" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT:   cir.store %{{.+}}, %[[THIS]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR-NEXT:   cir.store %{{.+}}, %[[K]] : !s32i, !cir.ptr<!s32i>

// LLVM:      define dso_local void @_ZN1S3setEi(ptr noundef nonnull align 4 dereferenceable(4) %{{.+}}, i32 noundef %{{.+}})
// LLVM:        %[[THIS:.*]] = alloca ptr, align 8
// LLVM-NEXT:   %[[K:.*]] = alloca i32, align 4
// LLVM-NEXT:   %[[Y:.*]] = alloca i32, align 4
// LLVM-NEXT:   store ptr %{{.+}}, ptr %[[THIS]], align 8
// LLVM-NEXT:   store i32 %{{.+}}, ptr %[[K]], align 4
// LLVM:        store i32 %{{.+}}, ptr %[[Y]], align 4

int S::pick(int k) {
  if (k)
    return m;
  {
    int w = k;
    use(&w);
  }
  return k;
}

// CIR:      cir.func {{.*}} @_ZN1S4pickEi(
// CIR-NEXT:   %[[RETVAL:.*]] = cir.alloca "__retval" align(4) : !cir.ptr<!s32i>
// CIR-NEXT:   %[[THIS:.*]] = cir.alloca "this" align(8) init : !cir.ptr<!cir.ptr<!rec_S>>
// CIR-NEXT:   %[[K:.*]] = cir.alloca "k" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT:   cir.store %{{.+}}, %[[THIS]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>
// CIR-NEXT:   cir.store %{{.+}}, %[[K]] : !s32i, !cir.ptr<!s32i>

// LLVM:      define dso_local noundef i32 @_ZN1S4pickEi(ptr noundef nonnull align 4 dereferenceable(4) %{{.+}}, i32 noundef %{{.+}})
// LLVM:        %[[RETVAL:.*]] = alloca i32, align 4
// LLVM-NEXT:   %[[THIS:.*]] = alloca ptr, align 8
// LLVM-NEXT:   %[[K:.*]] = alloca i32, align 4
// LLVM-NEXT:   %[[W:.*]] = alloca i32, align 4
// LLVM-NEXT:   store ptr %{{.+}}, ptr %[[THIS]], align 8
// LLVM-NEXT:   store i32 %{{.+}}, ptr %[[K]], align 4
// LLVM:        store i32 %{{.+}}, ptr %[[W]], align 4
// LLVM:        store i32 %{{.+}}, ptr %[[RETVAL]], align 4

struct Big {
  int a[8];
};

Big makeBig(int v) {
  Big b;
  {
    int z = v;
    use(&z);
  }
  b.a[0] = v;
  return b;
}

// CIR:      cir.func {{.*}} @_Z7makeBigi(
// CIR-NEXT:   %[[V:.*]] = cir.alloca "v" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT:   cir.store %{{.+}}, %[[V]] : !s32i, !cir.ptr<!s32i>

// LLVM:      define dso_local void @_Z7makeBigi(ptr dead_on_unwind noalias writable sret(%struct.Big) align 4 %{{.+}}, i32 noundef %{{.+}})
// LLVM:        %[[V:.*]] = alloca i32, align 4
// LLVM-NEXT:   %[[Z:.*]] = alloca i32, align 4
// LLVM-NEXT:   store i32 %{{.+}}, ptr %[[V]], align 4
// LLVM:        store i32 %{{.+}}, ptr %[[Z]], align 4

Big makeBig2(int v) {
  Big b;
  {
    int z = v;
    use(&z);
  }
  if (v)
    return Big{};
  b.a[0] = v;
  return b;
}

// CIR:      cir.func {{.*}} @_Z8makeBig2i(
// CIR-NEXT:   %[[V:.*]] = cir.alloca "v" align(4) init : !cir.ptr<!s32i>
// CIR-NEXT:   %[[B:.*]] = cir.alloca "b" align(4) : !cir.ptr<!rec_Big>
// CIR-NEXT:   cir.store %{{.+}}, %[[V]] : !s32i, !cir.ptr<!s32i>

// LLVM:      define dso_local void @_Z8makeBig2i(ptr dead_on_unwind noalias writable sret(%struct.Big) align 4 %{{.+}}, i32 noundef %{{.+}})
// LLVM:        %[[V:.*]] = alloca i32, align 4
// LLVM-NEXT:   %[[B:.*]] = alloca %struct.Big, align 4
// LLVM-NEXT:   %[[Z:.*]] = alloca i32, align 4
// LLVM-NEXT:   store i32 %{{.+}}, ptr %[[V]], align 4
// LLVM:        store i32 %{{.+}}, ptr %[[Z]], align 4
