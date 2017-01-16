// RUN: %clang_cc1 -triple armv7l-unknown-linux-gnueabihf -emit-llvm -O1 -disable-llvm-passes -std=c++03 %s -o - | FileCheck %s

// This test should not to generate llvm.lifetime.start/llvm.lifetime.end for
// f function because all temporary objects in this function are used for the
// final result

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
// CHECK: [[T1:%.*]] = alloca %class.T, align 4
// CHECK: [[T2:%.*]] = alloca %class.T, align 4
// CHECK: [[T3:%.*]] = alloca %class.T, align 4
// CHECK: [[T4:%.*]] = call %class.T* @_ZN1TC1EPKc(%class.T* [[T1]], i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str, i32 0, i32 0))
// CHECK: [[T5:%.*]] = call %class.T* @_ZN1TC1E1S(%class.T* [[T2]], [2 x i32] %{{.*}})
// CHECK: call void @_ZNK1T6concatERKS_(%class.T* sret [[T3]], %class.T* [[T1]], %class.T* dereferenceable(16) [[T2]])
// CHECK: [[T6:%.*]] = call i8* @_ZNK1T3strEv(%class.T* [[T3]])
// CHECK: ret i8* [[T6]]

  return T("[").concat(T(s)).str();
}
