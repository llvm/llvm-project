// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o -  | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -x c++ -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o -  | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM
// RUN: %clang_cc1 -x c++ -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s -check-prefix=LLVM

typedef struct {
    int a;
    int b;
} Data;

typedef int (*fun_t)(Data* d);

struct A;
typedef int (*fun_typ)(struct A*);

typedef struct A {
  fun_typ fun;  
} A;

// CIR: !ty_22A22 = !cir.struct<struct "A" {!cir.ptr<!cir.func<!cir.int<s, 32> (!cir.ptr<!cir.struct<struct "A">>)>>} #cir.record.decl.ast>
A a = {(fun_typ)0};

int extract_a(Data* d) {
    return d->a;
}

// CIR: cir.func {{@.*foo.*}}(%arg0: !cir.ptr<!ty_22Data22>
// CIR:   [[TMP0:%.*]] = cir.alloca !cir.ptr<!ty_22Data22>, cir.ptr <!cir.ptr<!ty_22Data22>>, ["d", init]
// CIR:   [[TMP1:%.*]] = cir.alloca !s32i, cir.ptr <!s32i>, ["__retval"]
// CIR:   [[TMP2:%.*]] = cir.alloca !cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22Data22>)>>, cir.ptr <!cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22Data22>)>>>, ["f", init]
// CIR:   cir.store %arg0, [[TMP0]] : !cir.ptr<!ty_22Data22>, cir.ptr <!cir.ptr<!ty_22Data22>>
// CIR:   [[TMP3:%.*]] = cir.const(#cir.ptr<null> : !cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22Data22>)>>) : !cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22Data22>)>>
// CIR:   cir.store [[TMP3]], [[TMP2]] : !cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22Data22>)>>, cir.ptr <!cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22Data22>)>>>
// CIR:   [[TMP4:%.*]] = cir.get_global {{@.*extract_a.*}} : cir.ptr <!cir.func<!s32i (!cir.ptr<!ty_22Data22>)>>
// CIR:   cir.store [[TMP4]], [[TMP2]] : !cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22Data22>)>>, cir.ptr <!cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22Data22>)>>>
// CIR:   [[TMP5:%.*]] = cir.load [[TMP2]] : cir.ptr <!cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22Data22>)>>>, !cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22Data22>)>>
// CIR:   [[TMP6:%.*]] = cir.load [[TMP0]] : cir.ptr <!cir.ptr<!ty_22Data22>>, !cir.ptr<!ty_22Data22>
// CIR:   [[TMP7:%.*]] = cir.call [[TMP5]]([[TMP6]]) : (!cir.ptr<!cir.func<!s32i (!cir.ptr<!ty_22Data22>)>>, !cir.ptr<!ty_22Data22>) -> !s32i
// CIR:   cir.store [[TMP7]], [[TMP1]] : !s32i, cir.ptr <!s32i>

// LLVM: define i32 {{@.*foo.*}}(ptr %0)
// LLVM:   [[TMP1:%.*]] = alloca ptr, i64 1
// LLVM:   [[TMP2:%.*]] = alloca i32, i64 1
// LLVM:   [[TMP3:%.*]] = alloca ptr, i64 1
// LLVM:   store ptr %0, ptr [[TMP1]]
// LLVM:   store ptr null, ptr [[TMP3]]
// LLVM:   store ptr {{@.*extract_a.*}}, ptr [[TMP3]]
// LLVM:   [[TMP4:%.*]] = load ptr, ptr [[TMP3]]
// LLVM:   [[TMP5:%.*]] = load ptr, ptr [[TMP1]]
// LLVM:   [[TMP6:%.*]] = call i32 [[TMP4]](ptr [[TMP5]])
// LLVM:   store i32 [[TMP6]], ptr [[TMP2]]
int foo(Data* d) {
    fun_t f = 0;
    f = extract_a;
    return f(d);
}

// CIR:  cir.func private {{@.*test.*}}() -> !cir.ptr<!cir.func<!void ()>>
// CIR:  cir.func {{@.*bar.*}}()
// CIR:    [[RET:%.*]] = cir.call {{@.*test.*}}() : () -> !cir.ptr<!cir.func<!void ()>>
// CIR:    cir.call [[RET]]() : (!cir.ptr<!cir.func<!void ()>>) -> ()
// CIR:    cir.return

// LLVM: declare {{.*}} ptr {{@.*test.*}}()
// LLVM: define void {{@.*bar.*}}()
// LLVM:   [[RET:%.*]] = call ptr {{@.*test.*}}()
// LLVM:   call void [[RET]]()
// LLVM:   ret void
void (*test(void))(void);
void bar(void) {
  test()();
}
