// DEFINE: %{analyze} = %clang_analyze_cc1 \
// DEFINE:   -analyzer-checker=core,unix.StdCLibraryFunctions \
// DEFINE:   -analyzer-config unix.StdCLibraryFunctions:ModelPOSIX=true

// RUN: %{analyze} -triple arm64-apple-darwin -verify=darwin %s
// RUN: %{analyze} -triple x86_64-unknown-linux-gnu -verify=linux %s

typedef unsigned long size_t;
typedef long off_t;
void *mmap(void *, size_t, int, int, int, off_t);

#define MAP_PRIVATE 0x0002
#define MAP_ANON    0x1000

// VM_MAKE_TAG on Darwin encodes a Mach VM memory tag in the top 8 bits.
// For tags >= 128 the result is a large negative signed integer.
#define VM_MAKE_TAG(tag) ((int)((unsigned)(tag) << 24))
#define VM_MEMORY_APPLICATION_SPECIFIC_1 240

void test_mmap_vm_make_tag(void) {
  // darwin-no-warning: no bound restriction on fd parameter on Darwin
  // linux-warning@+1 {{The 5th argument to 'mmap' is -268435456 but should be >= -1}}
  void *p = mmap(0, 4096, 0, MAP_ANON | MAP_PRIVATE,
                 VM_MAKE_TAG(VM_MEMORY_APPLICATION_SPECIFIC_1), 0);
}

void test_mmap_size_constraint(void) {
  void *p = mmap(0, 0, 0, MAP_ANON | MAP_PRIVATE, -1, 0);
  // darwin-warning@-1 {{The 2nd argument to 'mmap' is 0 but should be > 0}}
  // linux-warning@-2  {{The 2nd argument to 'mmap' is 0 but should be > 0}}
}
