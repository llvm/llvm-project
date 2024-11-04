// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

struct nested
{
  union {
    const char *single;
    const char *const *multi;
  } output;
};
static const char * const test[] = {
  "test",
};
const struct nested data[] = 
{
    {
        {
            .multi = test,
        },
    },
    {
        {
            .single = "hello",
        },
    },
};

// CIR: ![[ANON_TY:.+]] = !cir.struct<union "anon.0" {!cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CIR: ![[NESTED_TY:.+]] = !cir.struct<struct "nested" {![[ANON_TY]]
// CIR: cir.global constant external @data = #cir.const_array<[#cir.const_struct<{#cir.const_struct<{#cir.inactive_field : !cir.ptr<!s8i>, #cir.global_view<@test> : !cir.ptr<!cir.ptr<!s8i>>}> : ![[ANON_TY]]}> : ![[NESTED_TY:.+]], #cir.const_struct<{#cir.const_struct<{#cir.global_view<@".str"> : !cir.ptr<!s8i>, #cir.inactive_field : !cir.ptr<!cir.ptr<!s8i>>}> : ![[ANON_TY]]}> : ![[NESTED_TY:.+]]]> : !cir.array<![[NESTED_TY:.+]] x 2>
// LLVM: @data = constant [2 x {{.*}}]
