// RUN: %clang_cc1 -triple mipsel-unknown-linux-gnu -S -emit-llvm -o - %s | FileCheck -check-prefix=O32 %s
// RUN: %clang_cc1 -triple mips64el-unknown-linux-gnu -S -emit-llvm -o - %s  -target-abi n32 | FileCheck -check-prefix=N32-N64 %s
// RUN: %clang_cc1 -triple mips64el-unknown-linux-gnu -S -emit-llvm -o - %s  -target-abi n64 | FileCheck -check-prefix=N32-N64 %s

struct t1 {
  char t1[10];
};

struct t2 {
  char t2[20];
};

struct t3 {
  char t3[65];
};

extern struct t1 g1;
extern struct t2 g2;
extern struct t3 g3;
extern void f1(struct t1);
extern void f2(struct t2);
extern void f3(struct t3);

void f() {

// O32:  call void @f1(i32 inreg %{{[0-9]+}}, i32 inreg %{{[0-9]+}}, i16 inreg %{{[0-9]+}})
// O32:  call void @f2(%struct.t2* byval align 4 %{{.*}})
// O32:  call void @f3(%struct.t3* byval align 4 %{{.*}})

// N32-N64:  call void @f1(i64 inreg %{{[0-9]+}}, i16 inreg %{{[0-9]+}})
// N32-N64:  call void @f2(i64 inreg %{{[0-9]+}}, i64 inreg %{{[0-9]+}}, i32 inreg %{{[0-9]+}})
// N32-N64:  call void @f3(%struct.t3* byval align 8 %{{.*}})

  f1(g1);
  f2(g2);
  f3(g3);
}

