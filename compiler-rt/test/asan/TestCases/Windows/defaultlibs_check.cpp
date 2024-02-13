// RUN: %clang_cl_asan -Od %s -Fe%t /link /WX
// RUN: %clang_cl_asan -Od %s -Fe%t /link /WX /INFERASANLIBS:DEBUG

// Link with /WX under each configuration to ensure there are
// no warnings (ex: defaultlib mismatch, pragma detect mismatch)
// when linking.

int main() { return 0; }