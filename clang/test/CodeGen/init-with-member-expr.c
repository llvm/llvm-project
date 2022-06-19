// RUN: %clang_cc1 < %s -emit-llvm
struct test {
  int a;
};

extern struct test t;

int *b=&t.a;


// PR2049
typedef struct mark_header_tag {
 unsigned char mark[7];
} mark_header_t;
extern int foo();
int is_rar_archive(int fd) {
        const mark_header_t rar_hdr[2] = {{0x52, 0x61, 0x72, 0x21, 0x1a, 0x07, 0x00}, {'U', 'n', 'i', 'q', 'u', 'E', '!'}};
        foo(rar_hdr);

        return 0;
}

