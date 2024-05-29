/* RUN: %clang_msan -g %s -o %t
   RUN: %clang_msan -g %s -DBUILD_SO -fPIC -o %t-so.so -shared
   RUN: %run %t 2>&1 | FileCheck %s

   REQUIRES: glibc{{.*}}
*/

#define _GNU_SOURCE

#ifndef BUILD_SO
#include <assert.h>
#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>

typedef volatile long *(* get_t)();
get_t GetTls;

int main(int argc, char *argv[]) {
  char path[4096];
  snprintf(path, sizeof(path), "%s-so.so", argv[0]);
  int i;

  void *handle = dlopen(path, RTLD_LAZY);
  if (!handle) fprintf(stderr, "%s\n", dlerror());
  assert(handle != 0);
  GetTls = (get_t)dlsym(handle, "GetTls");
  assert(dlerror() == 0);

  {
    printf("Testing RTLD_DL_LINKMAP\n");
    fflush(stdout);

    Dl_info info;
    struct link_map *map_ptr;
    int ret = dladdr1(GetTls, &info, (void**)(&map_ptr), RTLD_DL_LINKMAP);
    assert(ret != 0);
    printf("fname: %s\n", info.dli_fname);
    printf("fbase: %p\n", info.dli_fbase);
    printf("sname: %s\n", info.dli_sname);
    // CHECK: sname: GetTls
    printf("saddr: %p\n", info.dli_saddr);

    assert(map_ptr != NULL);
    printf("map_ptr: %p\n", map_ptr);
    fflush(stdout);

    // Find the start of the link map
    while(map_ptr->l_prev != NULL) {
      fflush(stdout);
      map_ptr = map_ptr->l_prev;
    }

    fflush(stdout);
    while(map_ptr != NULL) {
      assert(map_ptr->l_name != NULL);
      printf("0x%lx: '%s', %p\n", map_ptr->l_addr, map_ptr->l_name, map_ptr->l_ld);
      fflush(stdout);
      map_ptr = map_ptr->l_next;
    }
    // CHECK: libc{{[\-]*.*}}.so
    // CHECK: dladdr1_test
  }

  // Test RTLD_DL_SYMENT

  {
    printf("Testing RTLD_DL_SYMENT\n");
    fflush(stdout);

    Dl_info info;
    ElfW(Sym) *sym;
    int ret = dladdr1(GetTls, &info, (void**)(&sym), RTLD_DL_SYMENT);
    assert(ret != 0);
    printf("fname: %s\n", info.dli_fname);
    printf("fbase: %p\n", info.dli_fbase);
    printf("sname: %s\n", info.dli_sname);
    // CHECK: sname: GetTls
    printf("saddr: %p\n", info.dli_saddr);

    printf("sym: %d %d %d %d %lu %lu\n",
           sym->st_name, sym->st_info, sym->st_other,
           sym->st_shndx, sym->st_value, sym->st_size);
    // CHECK: sym:
  }
  return 0;
}
#else  // BUILD_SO
long var;
long *GetTls() {
  return &var;
}
#endif
