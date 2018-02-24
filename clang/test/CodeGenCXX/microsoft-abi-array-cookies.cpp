// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 | FileCheck %s

struct ClassWithoutDtor {
  char x;
};

void check_array_no_cookies() {
// CHECK: define dso_local void @"\01?check_array_no_cookies@@YAXXZ"() [[NUW:#[0-9]+]]

// CHECK: call i8* @"\01??_U@YAPAXI@Z"(i32 42)
  ClassWithoutDtor *array = new ClassWithoutDtor[42];

// CHECK: call void @"\01??_V@YAXPAX@Z"(
  delete [] array;

}

struct ClassWithDtor {
  char x;
  ~ClassWithDtor() {}
};

void check_array_cookies_simple() {
// CHECK: define {{.*}} @"\01?check_array_cookies_simple@@YAXXZ"()

  ClassWithDtor *array = new ClassWithDtor[42];
// CHECK: [[ALLOCATED:%.*]] = call i8* @"\01??_U@YAPAXI@Z"(i32 46)
// 46 = 42 + size of cookie (4)
// CHECK: [[COOKIE:%.*]] = bitcast i8* [[ALLOCATED]] to i32*
// CHECK: store i32 42, i32* [[COOKIE]]
// CHECK: [[ARRAY:%.*]] = getelementptr inbounds i8, i8* [[ALLOCATED]], i32 4
// CHECK: bitcast i8* [[ARRAY]] to [[CLASS:%.*]]*

  delete [] array;
// CHECK: [[ARRAY_AS_CHAR:%.*]] = bitcast [[CLASS]]* {{%.*}} to i8*
// CHECK: getelementptr inbounds i8, i8* [[ARRAY_AS_CHAR]], i32 -4
}

struct __attribute__((aligned(8))) ClassWithAlignment {
  // FIXME: replace __attribute__((aligned(8))) with __declspec(align(8)) once
  // http://llvm.org/bugs/show_bug.cgi?id=12631 is fixed.
  int *x, *y;
  ~ClassWithAlignment() {}
};

void check_array_cookies_aligned() {
// CHECK: define {{.*}} @"\01?check_array_cookies_aligned@@YAXXZ"()
  ClassWithAlignment *array = new ClassWithAlignment[42];
// CHECK: [[ALLOCATED:%.*]] = call i8* @"\01??_U@YAPAXI@Z"(i32 344)
//   344 = 42*8 + size of cookie (8, due to alignment)
// CHECK: [[COOKIE:%.*]] = bitcast i8* [[ALLOCATED]] to i32*
// CHECK: store i32 42, i32* [[COOKIE]]
// CHECK: [[ARRAY:%.*]] = getelementptr inbounds i8, i8* [[ALLOCATED]], i32 8
// CHECK: bitcast i8* [[ARRAY]] to [[CLASS:%.*]]*

  delete [] array;
// CHECK: [[ARRAY_AS_CHAR:%.*]] = bitcast [[CLASS]]*
// CHECK: getelementptr inbounds i8, i8* [[ARRAY_AS_CHAR]], i32 -8
}

namespace PR23990 {
struct S {
  char x[42];
  void operator delete[](void *p, __SIZE_TYPE__);
  // CHECK-LABEL: define dso_local void @"\01?delete_s@PR23990@@YAXPAUS@1@@Z"(
  // CHECK: call void @"\01??_VS@PR23990@@SAXPAXI@Z"(i8* {{.*}}, i32 42)
};
void delete_s(S *s) { delete[] s; }
}

// CHECK: attributes [[NUW]] = { noinline nounwind{{.*}} }
