/*
RUN: rm -fr %t.profdir
RUN: %clang_profgen=%t.profdir/default_%m.profraw -o %t -O2 %s
RUN: %run %t  2>&1 | FileCheck %s --check-prefix=NO_EXIT_WRITE
RUN: llvm-profdata merge -o %t.profdata %t.profdir
RUN: %clang_profuse=%t.profdata -o - -S -emit-llvm %s | FileCheck %s  --check-prefix=PROF

NO_EXIT_WRITE: Profile data not written to file: already written
*/

int __llvm_profile_dump(void);
void __llvm_profile_reset_counters(void);
int foo(int);
int bar(int);
int skip(int);

int main(int argc, const char *argv[]) {
  int Ret = foo(0); /* region 1 */
  __llvm_profile_dump();

  /* not profiled -- cleared later. */
  skip(0);   /* skipped region */
  
  __llvm_profile_reset_counters();
  Ret += bar(0);  /* region 2 */
  __llvm_profile_dump();

  skip(1);

  __llvm_profile_reset_counters();
  /* foo's profile will be merged.  */
  foo(1);  /* region 3 */
  __llvm_profile_dump();

  return Ret;
}

__attribute__((noinline)) int foo(int X) {
  /* PROF: define {{.*}} @foo({{.*}}!prof ![[ENT:[0-9]+]]
     PROF: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD1:[0-9]+]]
  */
  return X <= 0 ? -X : X;
}

__attribute__((noinline)) int skip(int X) {
  /* PROF: define {{.*}} @skip(
     PROF: br i1 %{{.*}}, label %{{.*}}, label %{{[^,]+$}}
  */
  return X <= 0 ? -X : X;
}

__attribute__((noinline)) int bar(int X) {
  /* PROF-LABEL: define {{.*}} @bar(
     PROF: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[PD2:[0-9]+]]
  */
  return X <= 0 ? -X : X;
}

/*
PROF: ![[ENT]] = !{!"function_entry_count", i64 2}  
PROF: ![[PD1]] = !{!"branch_weights", i32 2, i32 2}
*/
