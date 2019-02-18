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
// CHECK: [[T1i8:%.*]] = bitcast %class.T* [[T1]] to i8*
// CHECK: call void @llvm.lifetime.start.p0i8(i64 16, i8* [[T1i8]])
//
// CHECK: [[T2i8:%.*]] = bitcast %class.T* [[T2]] to i8*
// CHECK: call void @llvm.lifetime.start.p0i8(i64 16, i8* [[T2i8]])
// CHECK: [[T4:%.*]] = call %class.T* @_ZN1TC1EPKc(%class.T* [[T2]], i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str, i32 0, i32 0))
//
// CHECK: [[T3i8:%.*]] = bitcast %class.T* [[T3]] to i8*
// CHECK: call void @llvm.lifetime.start.p0i8(i64 16, i8* [[T3i8]])
// CHECK: [[T5:%.*]] = call %class.T* @_ZN1TC1E1S(%class.T* [[T3]], [2 x i32] %{{.*}})
//
// CHECK: call void @_ZNK1T6concatERKS_(%class.T* sret [[T1]], %class.T* [[T2]], %class.T* dereferenceable(16) [[T3]])
// CHECK: [[T6:%.*]] = call i8* @_ZNK1T3strEv(%class.T* [[T1]])
//
// CHECK: call void @llvm.lifetime.end.p0i8(
// CHECK: call void @llvm.lifetime.end.p0i8(
// CHECK: call void @llvm.lifetime.end.p0i8(
// CHECK: ret i8* [[T6]]

  return T("[").concat(T(s)).str();
}

// CHECK: declare {{.*}}llvm.lifetime.start
// CHECK: declare {{.*}}llvm.lifetime.end
