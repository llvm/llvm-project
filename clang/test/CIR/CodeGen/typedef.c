// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s

void local_typedef() {
  typedef struct {int a;} Struct;
  Struct s;
}

//CHECK:  cir.func no_proto @local_typedef()
//CHECK:    {{.*}} = cir.alloca !ty_22Struct22, cir.ptr <!ty_22Struct22>, ["s"] {alignment = 4 : i64}
//CHECK:    cir.return
