# Libunwind customizations for linking with x86_64-fortanix-unknown-sgx Rust target.

## Description
### Initial Fork
Initial Fork has been made from 5.0 release of llvm (commit: 6a075b6de4)
### Detailed Description
#### Header files that we do not include for this target
1. pthread.h
#### Library that we do not link to for this target.
1. pthread (Locks used by libunwind is provided by rust stdlib for this target)

## Building unwind for rust-sgx target
### Generate Make files:
* `cd where you want to build libunwind`
* `mkdir build`
* `cd build`
* `cmake -DCMAKE_BUILD_TYPE="RELEASE" -DRUST_SGX=1 -G "Unix Makefiles" -DLLVM_ENABLE_WARNINGS=1 -DLIBUNWIND_ENABLE_WERROR=1 -DLIBUNWIND_ENABLE_PEDANTIC=0 -DLLVM_PATH=<path/to/llvm> <path/to/libunwind>`
* `"DEBUG"` could be used instead of `"RELEASE"` to enable debug logs of libunwind.

### Build:
* `make unwind_static`
* `build/lib/` will have the built library.
