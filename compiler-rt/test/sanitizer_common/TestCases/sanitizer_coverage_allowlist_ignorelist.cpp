// Tests -fsanitize-coverage-allowlist=allowlist.txt and
// -fsanitize-coverage-ignorelist=ignorelist.txt with libFuzzer-like coverage
// options

// REQUIRES: has_sancovcc,stable-runtime
// UNSUPPORTED: darwin
// XFAIL: ubsan,tsan
// XFAIL: android && asan

// RUN: DIR=%t_workdir
// RUN: rm -rf $DIR
// RUN: mkdir -p $DIR
// RUN: cd $DIR

// RUN: echo -e "src:*\nfun:*"     > al_all.txt
// RUN: echo -e ""                 > al_none.txt
// RUN: echo -e "src:%s\nfun:*"    > al_file.txt
// RUN: echo -e "src:*\nfun:*bar*" > al_bar.txt
// RUN: echo -e "src:*\nfun:*foo*" > al_foo.txt
// RUN: echo -e "src:*"            > bl_all.txt
// RUN: echo -e ""                 > bl_none.txt
// RUN: echo -e "src:%s"           > bl_file.txt
// RUN: echo -e "fun:*foo*"        > bl_foo.txt
// RUN: echo -e "fun:*bar*"        > bl_bar.txt

// Check inline-8bit-counters
// RUN: echo 'section "__sancov_cntrs"'                                                                             >  patterns.txt
// RUN: echo '%[0-9]\+ = load i8, ptr @__sancov_gen_' >> patterns.txt
// RUN: echo 'store i8 %[0-9]\+, ptr @__sancov_gen_'  >> patterns.txt
// RUN: echo '%[0-9]\+ = load i8, ptr getelementptr inbounds (\[[0-9]\+ x i8\], ptr @__sancov_gen_' >> patterns.txt
// RUN: echo 'store i8 %[0-9]\+, ptr getelementptr inbounds (\[[0-9]\+ x i8\], ptr @__sancov_gen_'  >> patterns.txt

// Check indirect-calls
// RUN: echo 'call void @__sanitizer_cov_trace_pc_indir'                                                            >> patterns.txt

// Check trace-cmp
// RUN: echo 'call void @__sanitizer_cov_trace_cmp4'                                                                >> patterns.txt

// Check pc-table
// RUN: echo 'section "__sancov_pcs"'                                                                               >> patterns.txt

// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table                                                                                      2>&1 |     grep -f patterns.txt | count 14
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_all.txt                                             2>&1 |     grep -f patterns.txt | count 14
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_none.txt                                            2>&1 | not grep -f patterns.txt
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_file.txt                                            2>&1 |     grep -f patterns.txt | count 14
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_foo.txt                                             2>&1 |     grep -f patterns.txt | count 9
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_bar.txt                                             2>&1 |     grep -f patterns.txt | count 5

// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table                                           -fsanitize-coverage-ignorelist=bl_all.txt   2>&1 | not grep -f patterns.txt
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_all.txt  -fsanitize-coverage-ignorelist=bl_all.txt   2>&1 | not grep -f patterns.txt
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_none.txt -fsanitize-coverage-ignorelist=bl_all.txt   2>&1 | not grep -f patterns.txt
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_file.txt -fsanitize-coverage-ignorelist=bl_all.txt   2>&1 | not grep -f patterns.txt
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_foo.txt  -fsanitize-coverage-ignorelist=bl_all.txt   2>&1 | not grep -f patterns.txt
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_bar.txt  -fsanitize-coverage-ignorelist=bl_all.txt   2>&1 | not grep -f patterns.txt

// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table                                           -fsanitize-coverage-ignorelist=bl_none.txt  2>&1 |     grep -f patterns.txt | count 14
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_all.txt  -fsanitize-coverage-ignorelist=bl_none.txt  2>&1 |     grep -f patterns.txt | count 14
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_none.txt -fsanitize-coverage-ignorelist=bl_none.txt  2>&1 | not grep -f patterns.txt
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_file.txt -fsanitize-coverage-ignorelist=bl_none.txt  2>&1 |     grep -f patterns.txt | count 14
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_foo.txt  -fsanitize-coverage-ignorelist=bl_none.txt  2>&1 |     grep -f patterns.txt | count 9
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_bar.txt  -fsanitize-coverage-ignorelist=bl_none.txt  2>&1 |     grep -f patterns.txt | count 5

// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table                                           -fsanitize-coverage-ignorelist=bl_file.txt  2>&1 | not grep -f patterns.txt
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_all.txt  -fsanitize-coverage-ignorelist=bl_file.txt  2>&1 | not grep -f patterns.txt
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_none.txt -fsanitize-coverage-ignorelist=bl_file.txt  2>&1 | not grep -f patterns.txt
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_file.txt -fsanitize-coverage-ignorelist=bl_file.txt  2>&1 | not grep -f patterns.txt
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_foo.txt  -fsanitize-coverage-ignorelist=bl_file.txt  2>&1 | not grep -f patterns.txt
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_bar.txt  -fsanitize-coverage-ignorelist=bl_file.txt  2>&1 | not grep -f patterns.txt

// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table                                           -fsanitize-coverage-ignorelist=bl_foo.txt   2>&1 |     grep -f patterns.txt | count 5
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_all.txt  -fsanitize-coverage-ignorelist=bl_foo.txt   2>&1 |     grep -f patterns.txt | count 5
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_none.txt -fsanitize-coverage-ignorelist=bl_foo.txt   2>&1 | not grep -f patterns.txt
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_file.txt -fsanitize-coverage-ignorelist=bl_foo.txt   2>&1 |     grep -f patterns.txt | count 5
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_foo.txt  -fsanitize-coverage-ignorelist=bl_foo.txt   2>&1 | not grep -f patterns.txt
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_bar.txt  -fsanitize-coverage-ignorelist=bl_foo.txt   2>&1 |     grep -f patterns.txt | count 5

// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table                                           -fsanitize-coverage-ignorelist=bl_bar.txt   2>&1 |     grep -f patterns.txt | count 9
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_all.txt  -fsanitize-coverage-ignorelist=bl_bar.txt   2>&1 |     grep -f patterns.txt | count 9
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_none.txt -fsanitize-coverage-ignorelist=bl_bar.txt   2>&1 | not grep -f patterns.txt
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_file.txt -fsanitize-coverage-ignorelist=bl_bar.txt   2>&1 |     grep -f patterns.txt | count 9
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_foo.txt  -fsanitize-coverage-ignorelist=bl_bar.txt   2>&1 |     grep -f patterns.txt | count 9
// RUN: %clangxx -O0 %s -S -o - -emit-llvm -fsanitize-coverage=inline-8bit-counters,indirect-calls,trace-cmp,pc-table -fsanitize-coverage-allowlist=al_bar.txt  -fsanitize-coverage-ignorelist=bl_bar.txt   2>&1 | not grep -f patterns.txt

// RUN: cd -
// RUN: rm -rf $DIR

// foo has 3 instrumentation points, 0 indirect call, 1 comparison point

// Expected results with patterns.txt when foo gets instrumentation with
// libFuzzer-like coverage options: 9 lines
//   inline-8bit-counters ->
//     @__sancov_gen_XX = private global [3 x i8] zeroinitializer, section "__sancov_cntrs"...
//     %XX = load i8, ptr @__sancov_gen_...
//     %XX = load i8, ptr getelementptr inbounds ([3 x i8], ptr @__sancov_gen_, i64 0, i64 1)...
//     %XX = load i8, ptr getelementptr inbounds ([3 x i8], ptr @__sancov_gen_, i64 0, i64 2)...
//     store i8 %XX, ptr @__sancov_gen_...
//     store i8 %XX, ptr getelementptr inbounds ([3 x i8], ptr @__sancov_gen_, i64 0, i64 1)...
//     store i8 %XX, ptr getelementptr inbounds ([3 x i8], ptr @__sancov_gen_, i64 0, i64 2)...
//   trace-cmp ->
//     call void @__sanitizer_cov_trace_cmp4(i32 %XX, i32 %XX)
//   pc-table ->
//     @__sancov_gen_XX = private constant [6 x ptr] ..., section "__sancov_pcs"...

bool foo(int *a, int *b) {
  if (*a == *b) {
    return true;
  }
  return false;
}

// bar has 1 instrumentation point, 1 indirect call, 0 comparison point

// Expected results with patterns.txt when bar gets instrumentation with
// libFuzzer-like coverage options: 5 lines
//   inline-8bit-counters ->
//     @__sancov_gen_XX = private global [1 x i8] zeroinitializer, section "__sancov_cntrs"...
//     %XX = load i8, ptr @__sancov_gen_.2, ...
//     store i8 %XX, ptr @__sancov_gen_.2, ...
//   indirect-calls ->
//     call void @__sanitizer_cov_trace_pc_indir(i64 %XX)
//   pc-table ->
//     @__sancov_gen_XX = private constant [2 x ptr] ..., section "__sancov_pcs"...

void bar(void (*f)()) { f(); }
