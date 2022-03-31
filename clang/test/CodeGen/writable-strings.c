// RUN: %clang_cc1 -emit-llvm -o - -fwritable-strings %s

int main(void) {
    char *str = "abc";
    str[0] = '1';
    printf("%s", str);
}

