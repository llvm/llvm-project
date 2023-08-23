file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/bc2h.c
  CONTENT
"#include <stdio.h>
int main(int argc, char **argv){
    FILE *ifp, *ofp;
    int c, i, l;
    if (argc != 4) return 1;
    ifp = fopen(argv[1], \"rb\");
    if (!ifp) return 1;
    i = fseek(ifp, 0, SEEK_END);
    if (i < 0) return 1;
    l = ftell(ifp);
    if (l < 0) return 1;
    i = fseek(ifp, 0, SEEK_SET);
    if (i < 0) return 1;
    ofp = fopen(argv[2], \"wb+\");
    if (!ofp) return 1;
    fprintf(ofp, \"#define %s_size %d\\n\\n\"
                 \"#if defined __GNUC__\\n\"
                 \"__attribute__((aligned (4096)))\\n\"
                 \"#elif defined _MSC_VER\\n\"
                 \"__declspec(align(4096))\\n\"
                 \"#endif\\n\"
                 \"static const unsigned char %s[%s_size+1] = {\",
                 argv[3], l,
                 argv[3], argv[3]);
    i = 0;
    while ((c = getc(ifp)) != EOF) {
        if (0 == (i&7)) fprintf(ofp, \"\\n   \");
        fprintf(ofp, \" 0x%02x,\", c);
        ++i;
    }
    fprintf(ofp, \" 0x00\\n};\\n\\n\");
    fclose(ifp);
    fclose(ofp);
    return 0;
}
")

add_executable(bc2h ${CMAKE_CURRENT_BINARY_DIR}/bc2h.c)
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_definitions(bc2h PRIVATE -D_CRT_SECURE_NO_WARNINGS)
endif()
