// RUN: %clang_cc1 -fsycl-is-device -fsyntax-only -verify %s
// expected-no-diagnostics

namespace cl {
namespace __spirv {

class OpTypeDeviceEvent;
extern __global OpTypeDeviceEvent* OpCreateUserEvent() noexcept;
extern void OpRetainEvent( __global OpTypeDeviceEvent * event ) noexcept;
extern void OpReleaseEvent( __global OpTypeDeviceEvent * event ) noexcept;

}
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([]() {
    __global cl::__spirv::OpTypeDeviceEvent *DE = cl::__spirv::OpCreateUserEvent();
    cl::__spirv::OpRetainEvent(DE);
    cl::__spirv::OpReleaseEvent(DE);
  });
  return 0;
}
