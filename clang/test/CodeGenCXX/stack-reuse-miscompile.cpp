// RUN: %clang_cc1 -triple armv7l-unknown-linux-gnueabihf -emit-llvm -O1 -disable-llvm-passes -std=c++03 %s -o - | FileCheck %s --implicit-check-not=llvm.lifetime

class S {
  char *ptr;
  unsigned int len;
};

class T {
  S left;
  S right;

public:
  T(const char s[]);
  T(S);

  T concat(const T &Suffix) const;
  const char * str() const;
};

const char * f(S s)
{
// It's essential that the lifetimes of all three T temporaries here are
// overlapping. They must all remain alive through the call to str().
//
// CHECK: [[T1:%.*]] = alloca %class.T, align 4
// CHECK: [[T2:%.*]] = alloca %class.T, align 4
// CHECK: [[T3:%.*]] = alloca %class.T, align 4
//
// FIXME: We could defer starting the lifetime of the return object of concat
// until the call.
// CHECK: call void @llvm.lifetime.start.p0(ptr [[T1]])
//
// CHECK: call void @llvm.lifetime.start.p0(ptr [[T2]])
// CHECK: [[T4:%.*]] = call noundef ptr @_ZN1TC1EPKc(ptr {{[^,]*}} [[T2]], ptr noundef @.str)
//
// CHECK: call void @llvm.lifetime.start.p0(ptr [[T3]])
// CHECK: [[T5:%.*]] = call noundef ptr @_ZN1TC1E1S(ptr {{[^,]*}} [[T3]], [2 x i32] %{{.*}})
//
// CHECK: call void @_ZNK1T6concatERKS_(ptr dead_on_unwind writable sret(%class.T) align 4 [[T1]], ptr {{[^,]*}} [[T2]], ptr noundef nonnull align 4 dereferenceable(16) [[T3]])
// CHECK: [[T6:%.*]] = call noundef ptr @_ZNK1T3strEv(ptr {{[^,]*}} [[T1]])
//
// CHECK: call void @llvm.lifetime.end.p0(
// CHECK: call void @llvm.lifetime.end.p0(
// CHECK: call void @llvm.lifetime.end.p0(
// CHECK: ret ptr [[T6]]

  return T("[").concat(T(s)).str();
}

// CHECK: declare {{.*}}llvm.lifetime.start
// CHECK: declare {{.*}}llvm.lifetime.end
