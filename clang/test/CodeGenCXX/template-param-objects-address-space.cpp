// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -std=c++20 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -std=c++20 %s -emit-llvm -o - | FileCheck %s --check-prefix=WITH-NONZERO-DEFAULT-AS

struct S { char buf[32]; };
template<S s> constexpr const char *begin() { return s.buf; }
template<S s> constexpr const char *end() { return s.buf + __builtin_strlen(s.buf); }
template<S s> constexpr const void *retval() { return &s; }
extern const void *callee(const S*);
template<S s> constexpr const void* observable_addr() { return callee(&s); }

// CHECK: [[HELLO:@_ZTAXtl1StlA32_cLc104ELc101ELc108ELc108ELc111ELc32ELc119ELc111ELc114ELc108ELc100EEEE]]
// WITH-NONZERO-DEFAULT-AS: [[HELLO:@_ZTAXtl1StlA32_cLc104ELc101ELc108ELc108ELc111ELc32ELc119ELc111ELc114ELc108ELc100EEEE]]
// CHECK-SAME: = linkonce_odr addrspace(1) constant { <{ [11 x i8], [21 x i8] }> } { <{ [11 x i8], [21 x i8] }> <{ [11 x i8] c"hello world", [21 x i8] zeroinitializer }> }, comdat

// CHECK: @p
// CHECK-SAME: addrspace(1) global ptr addrspacecast (ptr addrspace(1) [[HELLO]] to ptr)
// WITH-NONZERO-DEFAULT-AS: addrspace(1) global ptr addrspace(4) addrspacecast (ptr addrspace(1) [[HELLO]] to ptr addrspace(4))
const char *p = begin<S{"hello world"}>();

// CHECK: @q
// CHECK-SAME: addrspace(1) global ptr addrspacecast (ptr addrspace(1) getelementptr (i8, ptr addrspace(1) [[HELLO]], i64 11) to ptr)
// WITH-NONZERO-DEFAULT-AS: addrspace(1) global ptr addrspace(4) addrspacecast (ptr addrspace(1) getelementptr (i8, ptr addrspace(1) [[HELLO]], i64 11) to ptr addrspace(4))
const char *q = end<S{"hello world"}>();

const void *(*r)() = &retval<S{"hello world"}>;

// CHECK: @s
// CHECK-SAME: addrspace(1) global ptr null
// WITH-NONZERO-DEFAULT-AS: addrspace(1) global ptr addrspace(4) null
const void *s = observable_addr<S{"hello world"}>();

// CHECK: define linkonce_odr noundef ptr @_Z6retvalIXtl1StlA32_cLc104ELc101ELc108ELc108ELc111ELc32ELc119ELc111ELc114ELc108ELc100EEEEEPKvv()
// WITH-NONZERO-DEFAULT-AS: define linkonce_odr {{.*}} noundef ptr addrspace(4) @_Z6retvalIXtl1StlA32_cLc104ELc101ELc108ELc108ELc111ELc32ELc119ELc111ELc114ELc108ELc100EEEEEPKvv()
// CHECK: ret ptr addrspacecast (ptr addrspace(1) [[HELLO]] to ptr)
// WITH-NONZERO-DEFAULT-AS: ret ptr addrspace(4) addrspacecast (ptr addrspace(1) [[HELLO]] to ptr addrspace(4))

// CHECK: define linkonce_odr noundef ptr @_Z15observable_addrIXtl1StlA32_cLc104ELc101ELc108ELc108ELc111ELc32ELc119ELc111ELc114ELc108ELc100EEEEEPKvv()
// WITH-NONZERO-DEFAULT-AS: define linkonce_odr {{.*}} noundef ptr addrspace(4) @_Z15observable_addrIXtl1StlA32_cLc104ELc101ELc108ELc108ELc111ELc32ELc119ELc111ELc114ELc108ELc100EEEEEPKvv()
// CHECK: %call = call noundef ptr @_Z6calleePK1S(ptr noundef addrspacecast (ptr addrspace(1) [[HELLO]] to ptr))
// WITH-NONZERO-DEFAULT-AS: %call = call {{.*}} noundef{{.*}} ptr addrspace(4) @_Z6calleePK1S(ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) [[HELLO]] to ptr addrspace(4)))
// CHECK: declare noundef ptr @_Z6calleePK1S(ptr noundef)
// WITH-NONZERO-DEFAULT-AS: declare {{.*}} noundef ptr addrspace(4) @_Z6calleePK1S(ptr addrspace(4) noundef)
