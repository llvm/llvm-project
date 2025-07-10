// RUN: %clang_cc1  -fexperimental-pointer-field-protection=tagged -emit-llvm -o - %s | FileCheck %s

struct ClassWithTrivialCopy {
  ClassWithTrivialCopy();
  ~ClassWithTrivialCopy();
  void *a;
private:
  void *c;
};

// Make sure we don't emit memcpy for operator= and constuctors.
void make_trivial_copy(ClassWithTrivialCopy *s1, ClassWithTrivialCopy *s2) {
  *s1 = *s2;
  ClassWithTrivialCopy s3(*s2);
}

// CHECK-LABEL: define{{.*}} void @_Z17make_trivial_copyP20ClassWithTrivialCopyS0_
// CHECK-NOT: memcpy
// CHECK: ret void