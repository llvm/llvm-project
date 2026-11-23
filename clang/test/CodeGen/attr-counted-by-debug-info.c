// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -debug-info-kind=limited -fdebug-info-for-profiling -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -debug-info-kind=limited -fdebug-info-for-profiling -fsanitize=array-bounds -o - %s | FileCheck %s

// With -fsanitize=array-bounds, the bounds check for a subscript of a
// 'counted_by' pointer field reuses the field's address from the load of the
// base. Make sure that the load itself is emitted the same way with and
// without the sanitizer: it keeps the member's debug location, and the
// -fdebug-info-for-profiling pseudo variable for the struct base pointer.
// Both RUN lines share the same CHECK lines.

struct s {
  int n;
  int *buf __attribute__((counted_by(n)));
};

struct o {
  struct s *in;
};

struct s *get(void);

// CHECK-LABEL: define {{.*}} @nested(
// CHECK:         [[IN_FIELD:%.*]] = getelementptr inbounds nuw %struct.o, ptr {{%.*}}, i32 0, i32 0
// CHECK-NEXT:    [[IN:%.*]] = load ptr, ptr [[IN_FIELD]], align 8, !dbg [[NESTED_IN_LOC:![0-9]+]]
// CHECK-NEXT:      #dbg_value(ptr [[IN]], [[NESTED_PSEUDO:![0-9]+]], !DIExpression(), [[NESTED_IN_LOC]])
// CHECK-NEXT:    [[FIELD:%.*]] = getelementptr inbounds nuw %struct.s, ptr [[IN]], i32 0, i32 1, !dbg [[NESTED_LOC:![0-9]+]]
// CHECK-NEXT:    load ptr, ptr [[FIELD]], align 8, !dbg [[NESTED_LOC]]
int nested(struct o *q, int i) {
  return q->in->buf[i];
}

// CHECK-LABEL: define {{.*}} @call(
// CHECK:         [[CALL:%.*]] = call ptr @get(), !dbg [[CALL_GET_LOC:![0-9]+]]
// CHECK-NEXT:      #dbg_value(ptr [[CALL]], [[CALL_PSEUDO:![0-9]+]], !DIExpression(), [[CALL_GET_LOC]])
// CHECK-NEXT:    [[FIELD:%.*]] = getelementptr inbounds nuw %struct.s, ptr [[CALL]], i32 0, i32 1, !dbg [[CALL_LOC:![0-9]+]]
// CHECK-NEXT:    load ptr, ptr [[FIELD]], align 8, !dbg [[CALL_LOC]]
int call(int i) {
  return get()->buf[i];
}

// CHECK-LABEL: define {{.*}} @compound(
// CHECK:         [[IN_FIELD:%.*]] = getelementptr inbounds nuw %struct.o, ptr {{%.*}}, i32 0, i32 0
// CHECK-NEXT:    [[IN:%.*]] = load ptr, ptr [[IN_FIELD]], align 8, !dbg [[COMPOUND_IN_LOC:![0-9]+]]
// CHECK-NEXT:      #dbg_value(ptr [[IN]], [[COMPOUND_PSEUDO:![0-9]+]], !DIExpression(), [[COMPOUND_IN_LOC]])
// CHECK-NEXT:    [[FIELD:%.*]] = getelementptr inbounds nuw %struct.s, ptr [[IN]], i32 0, i32 1, !dbg [[COMPOUND_LOC:![0-9]+]]
// CHECK-NEXT:    load ptr, ptr [[FIELD]], align 8, !dbg [[COMPOUND_LOC]]
void compound(struct o *q, int i, int v) {
  q->in->buf[i] += v;
}

// The field load is located at the 'buf' member token in each function.
// CHECK-DAG: [[NESTED_PSEUDO]] = !DILocalVariable(scope: {{.*}}, flags: DIFlagArtificial)
// CHECK-DAG: [[NESTED_LOC]] = !DILocation(line: 29, column: 17,
// CHECK-DAG: [[CALL_PSEUDO]] = !DILocalVariable(scope: {{.*}}, flags: DIFlagArtificial)
// CHECK-DAG: [[CALL_LOC]] = !DILocation(line: 38, column: 17,
// CHECK-DAG: [[COMPOUND_PSEUDO]] = !DILocalVariable(scope: {{.*}}, flags: DIFlagArtificial)
// CHECK-DAG: [[COMPOUND_LOC]] = !DILocation(line: 48, column: 10,
