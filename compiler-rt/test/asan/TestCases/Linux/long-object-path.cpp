// RUN: mkdir -p %t.dir/a-long-directory-name-to-test-allocations-for-exceptions-in-_dl_lookup_symbol_x-since-glibc-2.27
// RUN: %clangxx_asan -g %s -o %t.dir/long-object-path
// RUN: %run %t.dir/a-*/../a-*/../a-*/../a-*/../a-*/../a-*/../a-*/../a-*/../long-object-path

int main(void) {
    return 0;
}
