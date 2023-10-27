// RUN: %clang_cc1 %s -fblocks -triple x86_64-apple-darwin -emit-llvm -o - | FileCheck %s

extern "C" {
extern "C" void *_Block_copy(const void *aBlock);
extern "C" void _Block_release(const void *aBlock);
}

class A {
public:
        int x;
        A(const A &o);
        A();
        virtual ~A();
        void hello() const;
};

int
main()
{
        A a;
        void (^c)(void) = ((__typeof(^{ a.hello(); }))_Block_copy((const void *)(^{ a.hello(); })));
        c();
        _Block_release((const void *)(c));
        return 0;
}

// CHECK-LABEL: define linkonce_odr hidden void @__copy_helper_block_
// CHECK: call void @_ZN1AC1ERKS_


// CHECK-LABEL:define linkonce_odr hidden void @__destroy_helper_block_
// CHECK: call void @_ZN1AD1Ev
