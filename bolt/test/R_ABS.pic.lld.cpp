/*
 * Check that we don't assert on a duplicate static relocation added by lld
 * against _Z6myfuncv. The same address has a dynamic relocation against it.
 *
 * This test uses the clang driver + lld and will only succeed on Linux systems
 * with libc available.
 * REQUIRES: system-linux
 *
 * RUN: %clang %cflags -fPIC -shared %s -o %t.so -Wl,-q -fuse-ld=lld
 * RUN: llvm-bolt %t.so -o %t.so.bolt --relocs
 */

unsigned long long myfunc();

unsigned long long (*myglobal)() = myfunc;

unsigned long long myfunc() {
  return reinterpret_cast<unsigned long long>(myglobal);
}
