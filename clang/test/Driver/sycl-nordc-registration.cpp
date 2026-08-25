// Check what a -fno-sycl-rdc object carries and when it is registered: build a
// two translation unit program, check that each object holds its own finalized
// device binary, then link and run the program to verify that both binaries
// reach the SYCL runtime.
// The program provides its own __sycl_register_lib/__sycl_unregister_lib, so
// -nolibsycl is used and no SYCL runtime or offload device is needed.

// REQUIRES: spirv-registered-target, system-linux, native

// RUN: rm -rf %t && mkdir -p %t
// RUN: %clangxx -fsycl -fno-sycl-rdc -c %s -o %t/main.o
// RUN: %clangxx -fsycl -fno-sycl-rdc -c \
// RUN:   %S/Inputs/sycl-nordc-registration-second-tu.cpp -o %t/second.o

// The section holds an offload binary (magic 0x10FF10AD) whose image is a
// finalized SPIR-V module (magic 0x07230203), both shown little endian by the
// hex dump.
// RUN: llvm-readelf --hex-dump=.sycl_fatbin %t/main.o | FileCheck %s \
// RUN:   --check-prefix=BINARY
// RUN: llvm-readelf --hex-dump=.sycl_fatbin %t/second.o | FileCheck %s \
// RUN:   --check-prefix=BINARY
// BINARY: 10ff10ad
// BINARY: 03022307

// Each binary offers the kernels of that translation unit.
// RUN: llvm-objcopy --dump-section=.sycl_fatbin=%t/main.bin %t/main.o /dev/null
// RUN: llvm-objcopy --dump-section=.sycl_fatbin=%t/second.bin %t/second.o /dev/null
// RUN: FileCheck %s --check-prefix=MAIN-BINARY --input-file=%t/main.bin
// RUN: FileCheck %s --check-prefix=SECOND-BINARY --input-file=%t/second.bin
// MAIN-BINARY-DAG: main_tu_kernel_name
// MAIN-BINARY-DAG: main_tu_other_kernel_name
// SECOND-BINARY-DAG: second_tu_kernel_name

// The finalized binary holds one entry.
// RUN: llvm-objdump --offloading %t/main.bin | FileCheck %s \
// RUN:   --check-prefix=MAIN-ENTRIES
// MAIN-ENTRIES:      OFFLOADING IMAGE [0]:
// MAIN-ENTRIES:      kind{{ *}}spir-v
// MAIN-ENTRIES:      arch{{ *$}}
// MAIN-ENTRIES-NOT:  OFFLOADING IMAGE

// Splitting a translation unit by kernel gives it an image
// per kernel, but they are merged into a single offload binary.
// RUN: %clangxx -fsycl -fno-sycl-rdc -fsycl-device-image-split=kernel -c %s \
// RUN:   -o %t/main.split.o
// RUN: llvm-objcopy --dump-section=.sycl_fatbin=%t/main.split.bin \
// RUN:   %t/main.split.o /dev/null
// RUN: llvm-objdump --offloading %t/main.split.bin | FileCheck %s \
// RUN:   --check-prefix=SPLIT-ENTRIES
// SPLIT-ENTRIES:      OFFLOADING IMAGE [0]:
// SPLIT-ENTRIES:      OFFLOADING IMAGE [1]:
// SPLIT-ENTRIES-NOT:  OFFLOADING IMAGE

// Between them the two images still offer both kernels of the translation unit,
// and the split program links and runs as the unsplit one did.
// RUN: FileCheck %s --check-prefix=MAIN-BINARY --input-file=%t/main.split.bin
// RUN: %clangxx -fsycl -nolibsycl %t/main.split.o %t/second.o -o %t/nordc-split
// RUN: %t/nordc-split | FileCheck %s --check-prefix=NORDC-OUT

// Both binaries are registered before the other static initializers of the
// program run, and unregistered after main returns.
// RUN: %clangxx -fsycl -nolibsycl %t/main.o %t/second.o -o %t/nordc
// RUN: %t/nordc | FileCheck %s --check-prefix=NORDC-OUT
// NORDC-OUT: registered binary 1
// NORDC-OUT-NEXT: registered binary 2
// NORDC-OUT-NEXT: static initializer sees 2
// NORDC-OUT-NEXT: main sees 2
// NORDC-OUT-NEXT: unregistered binary 1
// NORDC-OUT-NEXT: unregistered binary 2

// An RDC build of the same sources links the device code together, so a single
// binary is registered instead of one per translation unit.
// RUN: %clangxx -fsycl -fsycl-rdc -c %s -o %t/main.rdc.o
// RUN: %clangxx -fsycl -fsycl-rdc -c \
// RUN:   %S/Inputs/sycl-nordc-registration-second-tu.cpp -o %t/second.rdc.o
// RUN: %clangxx -fsycl -nolibsycl %t/main.rdc.o %t/second.rdc.o -o %t/rdc
// RUN: %t/rdc | FileCheck %s --check-prefix=RDC-OUT
// RDC-OUT: registered binary 1
// RDC-OUT-NEXT: static initializer sees 1
// RDC-OUT-NEXT: main sees 1
// RDC-OUT-NEXT: unregistered binary 1

#include <cstdio>
#include <cstdlib>
#include <cstring>

template <typename KernelName, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

struct main_tu_kernel_name;
struct main_tu_kernel {
  void operator()() const {}
};

[[clang::sycl_kernel_entry_point(main_tu_kernel_name)]]
void launch_main_tu_kernel(main_tu_kernel KernelFunc) {
  KernelFunc();
}

// A second kernel of the same translation unit to test that splitting works
// with no-rdc.
struct main_tu_other_kernel_name;
struct main_tu_other_kernel {
  void operator()() const {}
};

[[clang::sycl_kernel_entry_point(main_tu_other_kernel_name)]]
void launch_main_tu_other_kernel(main_tu_other_kernel KernelFunc) {
  KernelFunc();
}

void call_second_tu();

static int Registered = 0;
static int Unregistered = 0;

// Stand in for the SYCL runtime entry points the registration constructors and
// destructors call. Reject anything that is not an offload binary so that a
// malformed binary is not mistaken for a passing run.
extern "C" void __sycl_register_lib(void *Binary, size_t Size) {
  static const unsigned char OffloadBinaryMagic[] = {0x10, 0xFF, 0x10, 0xAD};
  if (Size <= sizeof(OffloadBinaryMagic) ||
      std::memcmp(Binary, OffloadBinaryMagic, sizeof(OffloadBinaryMagic)) != 0) {
    std::printf("registered something that is not an offload binary\n");
    std::exit(1);
  }
  std::printf("registered binary %d\n", ++Registered);
}

extern "C" void __sycl_unregister_lib(void *Binary, size_t Size) {
  std::printf("unregistered binary %d\n", ++Unregistered);
}

// A dynamic initializer of its own, so that the registration constructors have
// to be merged into a constructor list the translation unit already has.
struct RegistrationObserver {
  RegistrationObserver() {
    std::printf("static initializer sees %d\n", Registered);
  }
};
static RegistrationObserver Observer;

int main() {
  launch_main_tu_kernel(main_tu_kernel{});
  launch_main_tu_other_kernel(main_tu_other_kernel{});
  call_second_tu();
  std::printf("main sees %d\n", Registered);
  return 0;
}
