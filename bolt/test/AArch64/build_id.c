// This test checks that referencing build_id through GOT table
// would result in GOT access after disassembly, not directly
// to build_id address.

// RUN: %clang %cflags -fuse-ld=lld -Wl,-T,%S/Inputs/build_id.ldscript -Wl,-q \
// RUN:   -Wl,--no-relax -Wl,--build-id=sha1 %s -o %t.exe
// RUN: llvm-bolt -print-disasm --print-only=get_build_id %t.exe -o %t.bolt | \
// RUN:   FileCheck %s

// CHECK: adrp	[[REG:x[0-28]+]], __BOLT_got_zero
// CHECK: ldr x{{.*}}, [[[REG]], :lo12:__BOLT_got_zero{{.*}}]

struct build_id_note {
  char pad[16];
  char hash[20];
};

extern const struct build_id_note build_id_note;

__attribute__((noinline)) char get_build_id() { return build_id_note.hash[0]; }

int main() {
  get_build_id();
  return 0;
}
