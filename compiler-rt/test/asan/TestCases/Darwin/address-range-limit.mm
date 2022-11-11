// Regression test for https://code.google.com/p/address-sanitizer/issues/detail?id=368.

// RUN: %clangxx_asan %s -Wno-deprecated-declarations -flat_namespace -bundle -undefined suppress -o %t.bundle
// RUN: %clangxx_asan %s -Wno-deprecated-declarations -o %t -framework Foundation && not %run %t 2>&1 | FileCheck %s

// NSCreateObjectFileImageFromFile/NSLinkModule isn't available on iOS
// UNSUPPORTED: ios

#import <Foundation/Foundation.h>
#import <mach-o/dyld.h>

#include <string>

int main(int argc, char *argv[]) {
  for (int i = 0; i < 10; i++) {
    NSObjectFileImage im;

	std::string path = std::string(argv[0]) + ".bundle";
    NSObjectFileImageReturnCode rc =
        NSCreateObjectFileImageFromFile(path.c_str(), &im);
    if (rc != NSObjectFileImageSuccess) {
      fprintf(stderr, "Could not load bundle.\n");
      exit(-1);
    }

    NSModule handle = NSLinkModule(im, "a.bundle", 0);
    if (handle == 0) {
      fprintf(stderr, "Could not load bundle.\n");
      exit(-1);
    }
    printf("h: %p\n", handle);
  }

  char *ptr = (char *)malloc(10);
  ptr[10] = 'x';  // BOOM
}

// CHECK: AddressSanitizer: heap-buffer-overflow
// CHECK: WRITE of size 1
// CHECK: {{#0 .* in main}}
// CHECK: is located 0 bytes after 10-byte region
