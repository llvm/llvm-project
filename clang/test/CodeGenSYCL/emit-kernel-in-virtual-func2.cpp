// RUN: %clang -cc1 -triple spir64-unknown-linux-sycldevice  -std=c++11 -fsycl-is-device -emit-llvm -x c++ %s -o - | FileCheck %s

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
    kernelFunc();
}

template <class TAR>
void TT() {
    kernel_single_task<class PP>([]() {});
}

class BASE {
   public:
       virtual void initialize() {}
         virtual ~BASE();
};

template<class T>
class DERIVED : public BASE {
   public:
       void initialize() {
             TT<int>();
               }
};

int main() {
    BASE *Base = new DERIVED<int>;
      Base->initialize();
        delete Base;
}

// Ensure that the SPIR-Kernel function is actually emitted.
// CHECK: define spir_kernel void @PP
// CHECK: call spir_func void @_ZZ2TTIiEvvENKUlvE_clEv
// CHECK: define linkonce_odr spir_func void @_ZZ2TTIiEvvENKUlvE_clEv

