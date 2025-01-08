# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/test.s -o %t/test.o
# RUN: touch %t/empty.s; llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/empty.s -o %t/empty.o

# Check that `-allowable_client` generates LC_SUB_CLIENT.
# We create our .dylib in a `lib` subdirectory to make sure we test linking against the `.dylib` instead of the `.tbd` below.
# RUN: mkdir -p %t/lib; %lld -dylib -o %t/lib/liballowable_client.dylib %t/empty.o -allowable_client allowed -allowable_client also_allowed
# RUN: llvm-objdump --macho --all-headers %t/lib/liballowable_client.dylib | FileCheck %s
# CHECK:      LC_SUB_CLIENT
# CHECK-NEXT: cmdsize 24
# CHECK-NEXT: client allowed
# CHECK:      LC_SUB_CLIENT
# CHECK-NEXT: cmdsize 32
# CHECK-NEXT: client also_allowed

# Check linking against the .dylib we created above
# RUN: not %lld -o %t/test %t/test.o -L%t/lib -lallowable_client 2>&1 | FileCheck %s --check-prefix=NOTALLOWED-IMPLICIT
# RUN: not %lld -o %t/libtest_debug.exe %t/test.o -L%t/lib -lallowable_client 2>&1 | FileCheck %s --check-prefix=NOTALLOWED-IMPLICIT
# RUN: not %lld -o %t/test %t/test.o -L%t/lib -lallowable_client -client_name notallowed 2>&1 | FileCheck %s --check-prefix=NOTALLOWED-EXPLICIT
# RUN: %lld -o %t/test %t/test.o -L%t/lib -lallowable_client -client_name allowed
# RUN: %lld -o %t/test %t/test.o -L%t/lib -lallowable_client -client_name all
# RUN: %lld -o %t/all %t/test.o -L%t/lib -lallowable_client
# RUN: %lld -o %t/allowed %t/test.o -L%t/lib -lallowable_client
# RUN: %lld -o %t/liballowed_debug.exe %t/test.o -L%t/lib -lallowable_client

# Check linking against a .tbd
# RUN: not %lld -o %t/test %t/test.o -L%t -lallowable_client 2>&1 | FileCheck %s --check-prefix=NOTALLOWED-IMPLICIT
# RUN: not %lld -o %t/libtest_debug.exe %t/test.o -L%t -lallowable_client 2>&1 | FileCheck %s --check-prefix=NOTALLOWED-IMPLICIT
# RUN: not %lld -o %t/test %t/test.o -L%t -lallowable_client -client_name notallowed 2>&1 | FileCheck %s --check-prefix=NOTALLOWED-EXPLICIT
# RUN: %lld -o %t/test %t/test.o -L%t -lallowable_client -client_name allowed
# RUN: %lld -o %t/test %t/test.o -L%t -lallowable_client -client_name all
# RUN: %lld -o %t/all %t/test.o -L%t -lallowable_client
# RUN: %lld -o %t/allowed %t/test.o -L%t -lallowable_client
# RUN: %lld -o %t/liballowed_debug.exe %t/test.o -L%t -lallowable_client

# NOTALLOWED-IMPLICIT: error: cannot link directly with 'liballowable_client.dylib' because test is not an allowed client
# NOTALLOWED-EXPLICIT: error: cannot link directly with 'liballowable_client.dylib' because notallowed is not an allowed client

#--- test.s
.text
.globl _main
_main:
  ret

#--- liballowable_client.tbd
{
  "main_library": {
    "allowable_clients": [
      {
        "clients": [
          "allowed"
        ]
      }
    ],
    "compatibility_versions": [
      {
        "version": "0"
      }
    ],
    "current_versions": [
      {
        "version": "0"
      }
    ],
    "flags": [
      {
        "attributes": [
          "not_app_extension_safe"
        ]
      }
    ],
    "install_names": [
      {
        "name": "lib/liballowable_client.dylib"
      }
    ],
    "target_info": [
      {
        "min_deployment": "10.11",
        "target": "x86_64-macos"
      }
    ]
  },
  "tapi_tbd_version": 5
}
