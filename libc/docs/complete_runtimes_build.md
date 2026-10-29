(complete_runtimes_build)=

# Complete Runtimes Build for Linux

The first two steps are the same as a {ref}`full host build <full_host_build>`.

## Building a Simple Sysroot

### Step 1: Preparation

First, check out the demo branch in your llvm-project, then set up the
environment variables for your sysroot path.

```sh
cd llvm-project
gh pr checkout 234353
SYSROOT=$(readlink -f ~/sysroot)
```

### Step 2: Linux Headers

Next, install the Linux kernel headers into your sysroot. For this guide, we'll
copy the headers from the host system's `/usr/include` directory. This
includes `linux`, `asm-generic`, and the architecture-specific `asm`
headers.

```sh
# Create the include directory
mkdir -p $SYSROOT/usr/include

# Copy the header directories
cp -R /usr/include/linux $SYSROOT/usr/include/
cp -R /usr/include/asm-generic $SYSROOT/usr/include/
# Use -L to dereference the asm symlink and copy the actual files
cp -R -L /usr/include/asm $SYSROOT/usr/include/
```

If you instead want to build the headers from the kernel sources, you can run
the following commands:

```sh
# Create the include directory
mkdir -p $SYSROOT/usr/include

git clone --depth=1 git://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git /tmp/linux
make LLVM=1 INSTALL_HDR_PATH=$SYSROOT/usr -C /tmp/linux headers_install
```

The headers can be built to target non-host architectures by adding the
`ARCH={arm|arm64|i386}` to the above invocation of `make`.

:::{note}
For a more production-ready sysroot, you would typically download a specific
kernel version and install the headers using `make headers_install`
configured for the target architecture and installation path using the above
instructions.
:::

### Step 3: Build and Install Runtimes

For simplicity I'm going to assume you've already got a recent version of clang.
These instructions hard-code the clang version being 24 in a couple places.

```sh
# If your system clang isn't at least clang 22
CLANG_BIN=<path_to>/latest_clang/bin

cmake -G Ninja \
  -S runtimes \
  -B build-runtimes \
  -C libcxx/cmake/caches/Generic-llvm-libc.cmake \
  -DCMAKE_C_COMPILER=$CLANG_BIN/clang \
  -DCMAKE_CXX_COMPILER=$CLANG_BIN/clang++ \
  -DCMAKE_INSTALL_PREFIX=$SYSROOT/usr \
  -DLIBC_KERNEL_HEADERS=$SYSROOT/usr/include \
  -DCOMPILER_RT_INSTALL_PATH=$SYSROOT/usr/lib/clang/24
```

if you want to run the tests add
```sh
  -DLIBCXX_TEST_CONFIG="llvm-libc++-llvm-libc.cfg.in" \
  -DLIBCXXABI_TEST_CONFIG="llvm-libc++abi-llvm-libc.cfg.in" \
```

After configuring, build and install the necessary components:

```sh
ninja -C build-runtimes install
```

### Step 4: Build clang with the installed runtimes

```sh
cmake -G Ninja \
  -S llvm \
  -B build-clang-stage2 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$CLANG_BIN/clang \
  -DCMAKE_CXX_COMPILER=$CLANG_BIN/clang++ \
  -DCMAKE_SYSROOT=$SYSROOT \
  -DCMAKE_INSTALL_PREFIX=$SYSROOT/usr \
  -DCMAKE_EXE_LINKER_FLAGS="-resource-dir=$SYSROOT/usr/lib/clang/24 -stdlib=libc++ --rtlib=compiler-rt --unwindlib=libunwind -fuse-ld=lld -static" \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_ENABLE_LIBCXX=ON \
  -DLLVM_ENABLE_PIC=OFF

ninja -C build-clang-stage2 \
  install-clang install-lld install-clang-resource-headers \
  install-llvm-ar install-llvm-ranlib
```

### Step 5: Configure the Compiler

To make using the new toolchain easier, you can create a Clang configuration
file. This allows you to avoid passing long command line arguments every time
you compile a program.

```sh
cat << 'EOF' > $SYSROOT/usr/bin/x86_64-unknown-linux-gnu.cfg
--sysroot=<CFGDIR>/../..
-stdlib=libc++
-rtlib=compiler-rt
--unwindlib=libunwind
-fuse-ld=lld
-static
EOF
```

### Step 6: Verification

You can now use your newly built toolchain by running your compiler.

```c
// hello.c
#include <stdio.h>
int main() {
   printf("Hello, World!\n");
   return 0;
}
```

Compile and run the example:

```sh
$SYSROOT/usr/bin/clang hello.c
./a.out
```

### The All-in-One Command

This is so I can copy/paste the whole thing at once.

```sh
SYSROOT=$(readlink -f ~/sysroot)
CLANG_BIN=$(readlink -f ~/latest_clang/bin)
LINUX_HEADERS=$(readlink -f ~/linux_headers/include)

mkdir -p $SYSROOT/usr/include
cp -r $LINUX_HEADERS $SYSROOT/usr/

cmake -G Ninja \
  -S runtimes \
  -B build-runtimes \
  -C libcxx/cmake/caches/Generic-llvm-libc.cmake \
  -DCMAKE_C_COMPILER=$CLANG_BIN/clang \
  -DCMAKE_CXX_COMPILER=$CLANG_BIN/clang++ \
  -DCMAKE_INSTALL_PREFIX=$SYSROOT/usr \
  -DLIBC_KERNEL_HEADERS=$SYSROOT/usr/include \
  -DCOMPILER_RT_INSTALL_PATH=$SYSROOT/usr/lib/clang/24

ninja -C build-runtimes install

cmake -G Ninja \
  -S llvm \
  -B build-clang-stage2 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=$CLANG_BIN/clang \
  -DCMAKE_CXX_COMPILER=$CLANG_BIN/clang++ \
  -DCMAKE_SYSROOT=$SYSROOT \
  -DCMAKE_INSTALL_PREFIX=$SYSROOT/usr \
  -DCMAKE_EXE_LINKER_FLAGS="-resource-dir=$SYSROOT/usr/lib/clang/24 -stdlib=libc++ --rtlib=compiler-rt --unwindlib=libunwind -fuse-ld=lld -static" \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="X86" \
  -DLLVM_ENABLE_LIBCXX=ON \
  -DLLVM_ENABLE_PIC=OFF

ninja -C build-clang-stage2 \
  install-clang install-lld install-clang-resource-headers \
  install-llvm-ar install-llvm-ranlib

cat << 'EOF' > $SYSROOT/usr/bin/x86_64-unknown-linux-gnu.cfg
--sysroot=<CFGDIR>/../..
-stdlib=libc++
-rtlib=compiler-rt
--unwindlib=libunwind
-fuse-ld=lld
-static
EOF
```
