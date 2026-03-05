// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.og.ll %s

// Test in global context
#line 100 "test_file.cpp"
int global_line = __builtin_LINE();
// CIR: cir.global external @global_line = #cir.int<100> : !s32i
// LLVM: @global_line{{.*}} = global i32 100
// OGCG: @global_line{{.*}} = global i32 100

#line 15 "clang/test/CIR/CodeGen/source-loc-expr.cpp"
// Test __builtin_LINE
int test_builtin_LINE() {
  // CIR-LABEL: cir.func{{.*}} @{{.*}}test_builtin_LINE
  // CIR: %{{.*}} = cir.const #cir.int<25> : !u32i

  // LLVM-LABEL: @{{.*}}test_builtin_LINE
  // LLVM: store i32 25

  // OGCG-LABEL: @{{.*}}test_builtin_LINE
  // OGCG: ret i32 25
  return __builtin_LINE();
}

// Test __builtin_FILE
const char* test_builtin_FILE() {
  // CIR-LABEL: cir.func{{.*}} @{{.*}}test_builtin_FILE
  // CIR: %{{.*}} = cir.const #cir.global_view<@".str{{.*}}"> : !cir.ptr<!s8i>

  // LLVM-LABEL: @{{.*}}test_builtin_FILE
  // LLVM: store ptr @.str

  // OGCG-LABEL: @{{.*}}test_builtin_FILE
  // OGCG: ret ptr @.str
  return __builtin_FILE();
}

// Test __builtin_FUNCTION
const char* test_builtin_FUNCTION() {
  // CIR-LABEL: cir.func{{.*}} @{{.*}}test_builtin_FUNCTION
  // CIR: %{{.*}} = cir.const #cir.global_view<@".str{{.*}}"> : !cir.ptr<!s8i>

  // LLVM-LABEL: @{{.*}}test_builtin_FUNCTION
  // LLVM: store ptr @.str

  // OGCG-LABEL: @{{.*}}test_builtin_FUNCTION
  // OGCG: ret ptr @.str
  return __builtin_FUNCTION();
}

// Test __builtin_COLUMN
int test_builtin_COLUMN() {
  // CIR-LABEL: cir.func{{.*}} @{{.*}}test_builtin_COLUMN
  // The column number is the position of '__builtin_COLUMN'
  // CIR: %{{.*}} = cir.const #cir.int<10> : !u32i

  // LLVM-LABEL: @{{.*}}test_builtin_COLUMN
  // LLVM: store i32 10

  // OGCG-LABEL: @{{.*}}test_builtin_COLUMN
  // OGCG: ret i32 10
  return __builtin_COLUMN();
}

// Test default argument
int get_line(int l = __builtin_LINE()) {
  return l;
}

void test_default_arg() {
  // CIR-LABEL: cir.func{{.*}} @{{.*}}test_default_arg
  // The LINE should be from the call site, not the default argument definition
  #line 111
  int x = get_line();
  // CIR: %{{.*}} = cir.const #cir.int<111> : !u32i
  // CIR: %{{.*}} = cir.call @{{.*}}get_line{{.*}}({{.*}}) :

  // LLVM-LABEL: @{{.*}}test_default_arg
  // LLVM: call{{.*}} i32 @{{.*}}get_line{{.*}}(i32 111)

  // OGCG-LABEL: @{{.*}}test_default_arg
  // OGCG: call{{.*}} i32 @{{.*}}get_line{{.*}}(i32 {{.*}} 111)
}

#line 200 "lambda-test.cpp"
// Test in lambda (this tests that source location correctly captures context)
void test_in_lambda() {
  // CIR-LABEL: cir.func{{.*}} @{{.*}}test_in_lambda
  auto lambda = []() {
    return __builtin_LINE();
  };
  int x = lambda();

  // LLVM-LABEL: @{{.*}}test_in_lambda
  // LLVM: call{{.*}} i32 @{{.*}}

  // OGCG-LABEL: @{{.*}}test_in_lambda
  // OGCG: call{{.*}} i32 @{{.*}}
}

#line 214 "combined-test.cpp"
// Test multiple builtins in one expression
void test_combined() {
  // CIR-LABEL: cir.func{{.*}} @{{.*}}test_combined
  const char* file = __builtin_FILE();
  int line = __builtin_LINE();
  const char* func = __builtin_FUNCTION();
  // All should produce constants
  // CIR: cir.const
  // CIR: cir.const
  // CIR: cir.const

  // LLVM-LABEL: @{{.*}}test_combined
  // LLVM: store ptr @.str
  // LLVM: store i32 218
  // LLVM: store ptr @.str

  // OGCG-LABEL: @{{.*}}test_combined
  // OGCG: store ptr @.str
  // OGCG: store i32 218
  // OGCG: store ptr @.str
}
