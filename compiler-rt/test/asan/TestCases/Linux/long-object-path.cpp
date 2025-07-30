// RUN: mkdir -p %t.a-long-directory-name-to-test-allocations-for-exceptions-in-_dl_lookup_symbol_x-since-glibc-2.27
// RUN: %clangxx_asan -g %s -o %t.long-object-path
// RUN: %run %t.a-*/../a-*/../a-*/../a-*/../a-*/../a-*/../a-*/../a-*/../long-object-path

int main(void) {
    return 0;
}
