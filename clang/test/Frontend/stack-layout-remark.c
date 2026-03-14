// Check that backend stack layout diagnostics are working correctly with and
// without debug information, and when optimizations are enabled
//
// REQUIRES: x86-registered-target
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: %clang_cc1 %s -emit-codegen-only -triple x86_64-unknown-linux-gnu -target-cpu corei7 -Rpass-analysis=stack-frame-layout -o /dev/null  -O0  2>&1 | FileCheck %s --check-prefix=O0-NODEBUG
// RUN: %clang_cc1 %s -emit-codegen-only -triple x86_64-unknown-linux-gnu -target-cpu corei7 -Rpass-analysis=stack-frame-layout -o /dev/null  -O0  -debug-info-kind=constructor  -dwarf-version=5 -debugger-tuning=gdb 2>&1 | FileCheck %s --check-prefix=O0-DEBUG
// RUN: %clang_cc1 %s -emit-codegen-only -triple x86_64-unknown-linux-gnu -target-cpu corei7 -funwind-tables=2 -O3 -Rpass-analysis=stack-frame-layout   -debug-info-kind=constructor  -dwarf-version=5 -debugger-tuning=gdb -opt-record-file %t/stack-layout-remark.c.yml -opt-record-passes stack-frame-layout 2>&1 | FileCheck %s --check-prefix=O3-DEBUG
// RUN: cat %t/stack-layout-remark.c.yml | FileCheck %s --check-prefix=YAML

#define NULL (void*)0

extern void* allocate(unsigned size);
extern void deallocate(void* ptr);
extern int work(char *ary, int size);
extern int rand(void);

// Test YAML Ouput
// YAML: --- !Analysis
// YAML: Pass:            stack-frame-layout
// YAML: Name:            StackLayout
// YAML: DebugLoc:        { File: '{{.*}}stack-layout-remark.c',{{[[:space:]]*}}Line: [[# @LINE + 24]],
// YAML: Function:        foo
// YAML: Args:
// YAML:   - Offset:          '-40'
// YAML:   - Type:            Variable
// YAML:   - Align:           '16'
// YAML:   - Size:            '32'
// YAML:   - DataLoc:         'a @ {{.*}}stack-layout-remark.c:[[# @LINE + 19]]'
// YAML:   - DataLoc:         'f @ {{.*}}stack-layout-remark.c:[[# @LINE + 21]]'

//      O0-NODEBUG: Function: foo
// O0-NODEBUG-NEXT: Offset: [SP-40], Type: Variable, Align: 16, Size: 32
// O0-NODEBUG-NEXT: Offset: [SP-72], Type: Variable, Align: 16, Size: 32
//
//      O0-DEBUG: Function: foo
// O0-DEBUG-NEXT: Offset: [SP-40], Type: Variable, Align: 16, Size: 32
// O0-DEBUG-NEXT:     a @ {{.*}}stack-layout-remark.c:[[# @LINE + 10]]
// O0-DEBUG-NEXT: Offset: [SP-72], Type: Variable, Align: 16, Size: 32
// O0-DEBUG-NEXT:     f @ {{.*}}stack-layout-remark.c:[[# @LINE + 11]]

//      O3-DEBUG: Function: foo
// O3-DEBUG-NEXT: Offset: [SP-40], Type: Variable, Align: 16, Size: 32
// O3-DEBUG-NEXT:     a @ {{.*}}stack-layout-remark.c:[[# @LINE + 4]]
// O3-DEBUG-NEXT:     f @ {{.*}}stack-layout-remark.c:[[# @LINE + 6]]
void foo() {
  {
    char a[32] = {0};
    work(a, sizeof(a));
  }
  char f[32] = {0};
  work(f, sizeof(f));
}
//      O0-NODEBUG: Function: bar
// O0-NODEBUG-NEXT: Offset: [SP-40], Type: Variable, Align: 16, Size: 32
// O0-NODEBUG-NEXT: Offset: [SP-72], Type: Variable, Align: 16, Size: 32

//      O0-DEBUG: Function: bar
// O0-DEBUG-NEXT: Offset: [SP-40], Type: Variable, Align: 16, Size: 32
// O0-DEBUG-NEXT:     f @ {{.*}}stack-layout-remark.c:[[# @LINE + 10]]
// O0-DEBUG-NEXT: Offset: [SP-72], Type: Variable, Align: 16, Size: 32
// O0-DEBUG-NEXT:     a @ {{.*}}stack-layout-remark.c:[[# @LINE + 10]]

//      O3-DEBUG: Function: bar
// O3-DEBUG-NEXT: Offset: [SP-40], Type: Variable, Align: 16, Size: 32
// O3-DEBUG-NEXT:     f @ {{.*}}stack-layout-remark.c:[[# @LINE + 4]]
// O3-DEBUG-NEXT: Offset: [SP-72], Type: Variable, Align: 16, Size: 32
// O3-DEBUG-NEXT:     a @ {{.*}}stack-layout-remark.c:[[# @LINE + 4]]
void bar() {
  char f[32] = {0};
  {
    char a[32] = {0};
    work(a, sizeof(a));
  }
  work(f, sizeof(f));
}

struct Array {
  int *data;
  int size;
};

struct Result {
  struct Array *data;
  int sum;
};

//      O0-NODEBUG: Function: cleanup_array
// O0-NODEBUG-NEXT: Offset: [SP-8], Type: Variable, Align: 8, Size: 8

//      O0-DEBUG: Function: cleanup_array
// O0-DEBUG-NEXT: Offset: [SP-8], Type: Variable, Align: 8, Size: 8
// O0-DEBUG-NEXT:     a @ {{.*}}stack-layout-remark.c:[[# @LINE + 5]]

//      O3-DEBUG: Function: cleanup_array
//      O3-DEBUG: Function: cleanup_result
// O3-DEBUG-NEXT: Offset: [SP-8], Type: Spill, Align: 16, Size: 8
void cleanup_array(struct Array *a) {
  if (!a)
    return;
  if (!a->data)
    return;
  deallocate(a->data);
}

//      O0-NODEBUG: Function: cleanup_result
// O0-NODEBUG-NEXT: Offset: [SP-8], Type: Variable, Align: 8, Size: 8

//      O0-DEBUG: Function: cleanup_result
// O0-DEBUG-NEXT: Offset: [SP-8], Type: Variable, Align: 8, Size: 8
// O0-DEBUG-NEXT:     res @ {{.*}}stack-layout-remark.c:[[# @LINE + 1]]
void cleanup_result(struct Result *res) {
  if (!res)
    return;
  if (!res->data)
    return;
  cleanup_array(res->data);
  deallocate(res->data);
}

extern void use_dot_vector(struct Array *data);

//      O0-NODEBUG: Function: do_work
// O0-NODEBUG-NEXT: Offset: [SP-4], Type: Variable, Align: 4, Size: 4
// O0-NODEBUG-NEXT: Offset: [SP-16], Type: Variable, Align: 8, Size: 8
// O0-NODEBUG-NEXT: Offset: [SP-24], Type: Variable, Align: 8, Size: 8
// O0-NODEBUG-NEXT: Offset: [SP-32], Type: Variable, Align: 8, Size: 8
// O0-NODEBUG-NEXT: Offset: [SP-36], Type: Variable, Align: 4, Size: 4
// O0-NODEBUG-NEXT: Offset: [SP-48], Type: Variable, Align: 8, Size: 8
// O0-NODEBUG-NEXT: Offset: [SP-52], Type: Variable, Align: 4, Size: 4
// O0-NODEBUG-NEXT: Offset: [SP-56], Type: Variable, Align: 4, Size: 4

//      O0-DEBUG: Function: do_work
// O0-DEBUG-NEXT: Offset: [SP-4], Type: Variable, Align: 4, Size: 4
// O0-DEBUG-NEXT: Offset: [SP-16], Type: Variable, Align: 8, Size: 8
// O0-DEBUG-NEXT:     A @ {{.*}}stack-layout-remark.c:[[# @LINE + 20]]
// O0-DEBUG-NEXT: Offset: [SP-24], Type: Variable, Align: 8, Size: 8
// O0-DEBUG-NEXT:     B @ {{.*}}stack-layout-remark.c:[[# @LINE + 18]]
// O0-DEBUG-NEXT: Offset: [SP-32], Type: Variable, Align: 8, Size: 8
// O0-DEBUG-NEXT:     out @ {{.*}}stack-layout-remark.c:[[# @LINE + 16]]
// O0-DEBUG-NEXT: Offset: [SP-36], Type: Variable, Align: 4, Size: 4
// O0-DEBUG-NEXT:     len @ {{.*}}stack-layout-remark.c:[[# @LINE + 19]]
// O0-DEBUG-NEXT: Offset: [SP-48], Type: Variable, Align: 8, Size: 8
// O0-DEBUG-NEXT:     AB @ {{.*}}stack-layout-remark.c:[[# @LINE + 18]]
// O0-DEBUG-NEXT: Offset: [SP-52], Type: Variable, Align: 4, Size: 4
// O0-DEBUG-NEXT:     sum @ {{.*}}stack-layout-remark.c:[[# @LINE + 32]]
// O0-DEBUG-NEXT: Offset: [SP-56], Type: Variable, Align: 4, Size: 4
// O0-DEBUG-NEXT:     i @ {{.*}}stack-layout-remark.c:[[# @LINE + 31]]

//      O3-DEBUG: Function: do_work
// O3-DEBUG-NEXT: Offset: [SP-8], Type: Spill, Align: 16, Size: 8
// O3-DEBUG-NEXT: Offset: [SP-16], Type: Spill, Align: 8, Size: 8
// O3-DEBUG-NEXT: Offset: [SP-24], Type: Spill, Align: 16, Size: 8
// O3-DEBUG-NEXT: Offset: [SP-32], Type: Spill, Align: 8, Size: 8
// O3-DEBUG-NEXT: Offset: [SP-40], Type: Spill, Align: 16, Size: 8
int do_work(struct Array *A, struct Array *B, struct Result *out) {
  if (!A || !B)
    return -1;
  if (A->size != B->size)
    return -1;
  const int len = A->size;
  struct Array *AB;
  if (out->data == NULL) {
    AB = (struct Array *)allocate(sizeof(struct Array));
    AB->data = NULL;
    AB->size = 0;
    out->data = AB;
  } else {
    AB = out->data;
  }

  if (AB->data)
    deallocate(AB->data);

  AB->data = (int *)allocate(len * sizeof(int));
  AB->size = len;

  int sum = 0;
  for (int i = 0; i < len; ++i) {
    AB->data[i] = A->data[i] * B->data[i];
    sum += AB->data[i];
  }
  return sum;
}

//      O0-NODEBUG: Function: gen_array
// O0-NODEBUG-NEXT: Offset: [SP-8], Type: Variable, Align: 8, Size: 8
// O0-NODEBUG-NEXT: Offset: [SP-12], Type: Variable, Align: 4, Size: 4
// O0-NODEBUG-NEXT: Offset: [SP-24], Type: Variable, Align: 8, Size: 8
// O0-NODEBUG-NEXT: Offset: [SP-28], Type: Variable, Align: 4, Size: 4

//      O0-DEBUG: Function: gen_array
// O0-DEBUG-NEXT: Offset: [SP-8], Type: Variable, Align: 8, Size: 8
// O0-DEBUG-NEXT: Offset: [SP-12], Type: Variable, Align: 4, Size: 4
// O0-DEBUG-NEXT:     size @ {{.*}}stack-layout-remark.c:[[# @LINE + 10]]
// O0-DEBUG-NEXT: Offset: [SP-24], Type: Variable, Align: 8, Size: 8
// O0-DEBUG-NEXT:     res @ {{.*}}stack-layout-remark.c:[[# @LINE + 11]]
// O0-DEBUG-NEXT: Offset: [SP-28], Type: Variable, Align: 4, Size: 4
// O0-DEBUG-NEXT:     i @ {{.*}}stack-layout-remark.c:[[# @LINE + 13]]

//      O3-DEBUG: Function: gen_array
// O3-DEBUG-NEXT: Offset: [SP-8], Type: Spill, Align: 16, Size: 8
// O3-DEBUG-NEXT: Offset: [SP-16], Type: Spill, Align: 8, Size: 8
// O3-DEBUG-NEXT: Offset: [SP-24], Type: Spill, Align: 16, Size: 8
struct Array *gen_array(int size) {
  if (size < 0)
    return NULL;
  struct Array *res = (struct Array *)allocate(sizeof(struct Array));
  res->size = size;
  res->data = (int *)allocate(size * sizeof(int));

  for (int i = 0; i < size; ++i) {
    res->data[i] = rand();
  }

  return res;
}

// YAML: --- !Analysis
// YAML: Pass:            stack-frame-layout
// YAML: Name:            StackLayout
// YAML: DebugLoc:        { File: '{{.*}}stack-layout-remark.c',{{[[:space:]]*}}Line: [[# @LINE + 59]],
// YAML: Function:        caller
// YAML: Args:
// YAML:   - Offset:          '-8'
// YAML:   - Type:            Spill
// YAML:   - Align:           '16'
// YAML:   - Size:            '8'
// YAML:   - Offset:          '-16'
// YAML:   - Type:            Spill
// YAML:   - Align:           '8'
// YAML:   - Size:            '8'
// YAML:   - Offset:          '-24'
// YAML:   - Type:            Spill
// YAML:   - Align:           '16'
// YAML:   - Size:            '8'
// YAML:   - Offset:          '-32'
// YAML:   - Type:            Spill
// YAML:   - Align:           '8'
// YAML:   - Size:            '8'
// YAML:   - Offset:          '-40'
// YAML:   - Type:            Spill
// YAML:   - Align:           '16'
// YAML:   - Size:            '8'
// YAML:   - Offset:          '-48'
// YAML:   - Type:            Spill
// YAML:   - Align:           '8'
// YAML:   - Size:            '8'

//      O0-NODEBUG: Function: caller
// O0-NODEBUG-NEXT: Offset: [SP-4], Type: Variable, Align: 4, Size: 4
// O0-NODEBUG-NEXT: Offset: [SP-8], Type: Variable, Align: 4, Size: 4
// O0-NODEBUG-NEXT: Offset: [SP-16], Type: Variable, Align: 8, Size: 8
// O0-NODEBUG-NEXT: Offset: [SP-24], Type: Variable, Align: 8, Size: 8
// O0-NODEBUG-NEXT: Offset: [SP-32], Type: Variable, Align: 8, Size: 8
// O0-NODEBUG-NEXT: Offset: [SP-36], Type: Variable, Align: 4, Size: 4
// O0-NODEBUG-NEXT: Offset: [SP-40], Type: Variable, Align: 4, Size: 4

//      O0-DEBUG: Function: caller
// O0-DEBUG-NEXT: Offset: [SP-4], Type: Variable, Align: 4, Size: 4
// O0-DEBUG-NEXT: Offset: [SP-8], Type: Variable, Align: 4, Size: 4
// O0-DEBUG-NEXT:     size @ {{.*}}stack-layout-remark.c:[[# @LINE + 20]]
// O0-DEBUG-NEXT: Offset: [SP-16], Type: Variable, Align: 8, Size: 8
// O0-DEBUG-NEXT:     A @ {{.*}}stack-layout-remark.c:[[# @LINE + 19]]
// O0-DEBUG-NEXT: Offset: [SP-24], Type: Variable, Align: 8, Size: 8
// O0-DEBUG-NEXT:     B @ {{.*}}stack-layout-remark.c:[[# @LINE + 18]]
// O0-DEBUG-NEXT: Offset: [SP-32], Type: Variable, Align: 8, Size: 8
// O0-DEBUG-NEXT:     res @ {{.*}}stack-layout-remark.c:[[# @LINE + 17]]
// O0-DEBUG-NEXT: Offset: [SP-36], Type: Variable, Align: 4, Size: 4
// O0-DEBUG-NEXT:     ret @ {{.*}}stack-layout-remark.c:[[# @LINE + 16]]
// O0-DEBUG-NEXT: Offset: [SP-40], Type: Variable, Align: 4, Size: 4
// O0-DEBUG-NEXT:     err @ {{.*}}stack-layout-remark.c:[[# @LINE + 16]]

//      O3-DEBUG: Function: caller
// O3-DEBUG-NEXT: Offset: [SP-8], Type: Spill, Align: 16, Size: 8
// O3-DEBUG-NEXT: Offset: [SP-16], Type: Spill, Align: 8, Size: 8
// O3-DEBUG-NEXT: Offset: [SP-24], Type: Spill, Align: 16, Size: 8
// O3-DEBUG-NEXT: Offset: [SP-32], Type: Spill, Align: 8, Size: 8
// O3-DEBUG-NEXT: Offset: [SP-40], Type: Spill, Align: 16, Size: 8
// O3-DEBUG-NEXT: Offset: [SP-48], Type: Spill, Align: 8, Size: 8
int caller() {
  const int size = 100;
  struct Array *A = gen_array(size);
  struct Array *B = gen_array(size);
  struct Result *res = (struct Result *)allocate(sizeof(struct Result));
  int ret = -1;

  int err = do_work(A, B, res);
  if (err == -1) {
    goto cleanup;
  }

  ret = res->sum;
  if (ret == -1)
    return caller();

  use_dot_vector(res->data);

cleanup:
  cleanup_array(A);
  cleanup_array(B);
  cleanup_result(res);

  return ret;
}

