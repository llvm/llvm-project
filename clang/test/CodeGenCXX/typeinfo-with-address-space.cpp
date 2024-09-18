// RUN: %clang_cc1 -I%S %s -triple amdgcn-amd-amdhsa -emit-llvm -o - | FileCheck %s -check-prefix=AS
// RUN: %clang_cc1 -I%S %s -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s -check-prefix=NO-AS
#include <typeinfo>

class A {
    virtual void f() = 0;
};

class B : A {
    void f() override;
};

// AS: @_ZTISt9type_info = external addrspace(1) constant ptr addrspace(1)
// NO-AS: @_ZTISt9type_info = external constant ptr
// AS: @_ZTIi = external addrspace(1) constant ptr addrspace(1)
// NO-AS: @_ZTIi = external constant ptr
// AS: @_ZTVN10__cxxabiv117__class_type_infoE = external addrspace(1) global [0 x ptr addrspace(1)]
// NO-AS: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// AS: @_ZTS1A = linkonce_odr addrspace(1) constant [3 x i8] c"1A\00", comdat, align 1
// NO-AS: @_ZTS1A = linkonce_odr constant [3 x i8] c"1A\00", comdat, align 1
// AS: @_ZTI1A = linkonce_odr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) getelementptr inbounds (ptr addrspace(1), ptr addrspace(1) @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr addrspace(1) @_ZTS1A }, comdat, align 8
// NO-AS: @_ZTI1A = linkonce_odr constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1A }, comdat, align 8
// AS: @_ZTIf = external addrspace(1) constant ptr addrspace(1)
// NO-AS: @_ZTIf = external constant ptr

unsigned long Fn(B& b) {
// AS: %call = call noundef zeroext i1 @_ZNKSt9type_infoeqERKS_(ptr {{.*}} addrspacecast (ptr addrspace(1) @_ZTISt9type_info to ptr), ptr {{.*}} %2)
// NO-AS: %call = call noundef zeroext i1 @_ZNKSt9type_infoeqERKS_(ptr {{.*}} @_ZTISt9type_info, ptr {{.*}} %2)
    if (typeid(std::type_info) == typeid(b))
        return 42;
// AS: %call2 = call noundef zeroext i1 @_ZNKSt9type_infoneERKS_(ptr {{.*}} addrspacecast (ptr addrspace(1) @_ZTIi to ptr), ptr {{.*}} %5)
// NO-AS: %call2 = call noundef zeroext i1 @_ZNKSt9type_infoneERKS_(ptr {{.*}} @_ZTIi, ptr {{.*}} %5)
    if (typeid(int) != typeid(b))
        return 1712;
// AS: %call5 = call noundef ptr @_ZNKSt9type_info4nameEv(ptr {{.*}} addrspacecast (ptr addrspace(1) @_ZTI1A to ptr))
// NO-AS: %call5 = call noundef ptr @_ZNKSt9type_info4nameEv(ptr {{.*}} @_ZTI1A)
// AS: %call7 = call noundef ptr @_ZNKSt9type_info4nameEv(ptr {{.*}} %8)
// NO-AS: %call7 = call noundef ptr @_ZNKSt9type_info4nameEv(ptr {{.*}} %8)
    if (typeid(A).name() == typeid(b).name())
        return 0;
// AS: %call11 = call noundef zeroext i1 @_ZNKSt9type_info6beforeERKS_(ptr {{.*}} %11, ptr {{.*}} addrspacecast (ptr addrspace(1) @_ZTIf to ptr))
// NO-AS:   %call11 = call noundef zeroext i1 @_ZNKSt9type_info6beforeERKS_(ptr {{.*}} %11, ptr {{.*}} @_ZTIf)
    if (typeid(b).before(typeid(float)))
        return 1;
// AS: %call15 = call noundef i64 @_ZNKSt9type_info9hash_codeEv(ptr {{.*}} %14)
// NO-AS: %call15 = call noundef i64 @_ZNKSt9type_info9hash_codeEv(ptr {{.*}} %14)
    return typeid(b).hash_code();
}
