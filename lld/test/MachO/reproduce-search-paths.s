# REQUIRES: x86
# UNSUPPORTED: system-windows

# RUN: rm -rf %t; mkdir -p %t/host %t/sdk1/%:t/host/Foo.framework %t/sdk2/%:t/host/Bar.framework
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-macos %s -o %t/foo.o
# RUN: %no-arg-lld -arch x86_64 -platform_version macos 11.0 11.0 -dylib -Z \
# RUN:     %t/foo.o -o %t/sdk1/%:t/host/libfoo.dylib
# RUN: cp %t/sdk1/%:t/host/libfoo.dylib %t/sdk1/%:t/host/Foo.framework/Foo
# RUN: cp %t/sdk1/%:t/host/libfoo.dylib %t/sdk2/%:t/host/libbar.dylib
# RUN: cp %t/sdk1/%:t/host/libfoo.dylib %t/sdk2/%:t/host/Bar.framework/Bar

## Libraries and frameworks exist in SDKs only.
# RUN: %no-arg-lld -arch x86_64 -platform_version macos 11.0 11.0 -dylib -Z \
# RUN:     -syslibroot %t/sdk1 -syslibroot %t/sdk2 -L %t/host -F %t/host \
# RUN:     -lfoo -lbar -framework Foo -framework Bar -o %t/out --reproduce %t/repro.tar
# RUN: cd %t; tar xf repro.tar
# RUN: FileCheck %s --check-prefix=SDK -DPATH=%:t < repro/response.txt
# SDK:      -L [[PATH]]/sdk1/[[PATH]]/host
# SDK-NEXT: -L [[PATH]]/sdk2/[[PATH]]/host
# SDK-NEXT: -F [[PATH]]/sdk1/[[PATH]]/host
# SDK-NEXT: -F [[PATH]]/sdk2/[[PATH]]/host
# RUN: rm -rf sdk1 sdk2
# RUN: cd repro; %no-arg-lld @response.txt

## Libraries and frameworks exist outside the SDK only.
## Absolute paths fall back to host directories; relative paths ignore SDK roots.
# RUN: mkdir -p %t/sdk/relative %t/fallback/Foo.framework %t/relative/Bar.framework
# RUN: %no-arg-lld -arch x86_64 -platform_version macos 11.0 11.0 -dylib -Z \
# RUN:     %t/foo.o -o %t/fallback/libfoo.dylib
# RUN: cp %t/fallback/libfoo.dylib %t/fallback/Foo.framework/Foo
# RUN: cp %t/fallback/libfoo.dylib %t/relative/libbar.dylib
# RUN: cp %t/fallback/libfoo.dylib %t/relative/Bar.framework/Bar
# RUN: cd %t
# RUN: %no-arg-lld -arch x86_64 -platform_version macos 11.0 11.0 -dylib -Z \
# RUN:     -syslibroot %t/sdk -L %t/fallback -F %t/fallback -L relative -F relative \
# RUN:     -lfoo -lbar -framework Foo -framework Bar -o out --reproduce fallback.tar
# RUN: tar xf fallback.tar
# RUN: FileCheck %s --check-prefix=FALLBACK -DPATH=%:t < fallback/response.txt
# FALLBACK:      -L [[PATH]]/fallback
# FALLBACK-NEXT: -F [[PATH]]/fallback
# FALLBACK-NEXT: -L [[PATH]]/relative
# FALLBACK-NEXT: -F [[PATH]]/relative
# RUN: rm -rf sdk relative fallback/libfoo.dylib fallback/Foo.framework
# RUN: cd fallback; %no-arg-lld @response.txt

.globl _foo
_foo:
  ret
