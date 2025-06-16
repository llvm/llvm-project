
namespace LIBC_NAMESPACE_DECL {
namespace internal {
    void utf32to8(char* out, wchar_t src) {
        char u, v, w, x, y, z;
        u = (src & (0xf << 20)) >> 20;
        v = (src & (0xf << 16)) >> 16;
        w = (src & (0xf << 12)) >> 12;
        x = (src & (0xf <<  8)) >>  8;
        y = (src & (0xf <<  4)) >>  4;
        z = (src & (0xf <<  0)) >>  0;

        if (src <= 0x7f) {
            // 0yyyzzzz
            out[0] = src;
        }
        else if (src <= 0x7ff) {
            // 110xxxyy 10yyzzzz
            out[0] = 0xC0 | (x << 2) | (y >> 2);
            out[1] = 0x80 | ((y & 0x3) << 4) | z;
        }
        else if (src <= 0xffff) {
            // 1110wwww 10xxxxyy 10yyzzzz
            out[0] = 0xE0 | w;
            out[1] = 0x80 | (x << 2) | (y >> 2);
            out[2] = 0x80 | ((y & 0x3) << 4) | z;
        }
        else if (src <= 0x10ffff) {
            // 11110uvv 10vvwwww 10xxxxyy 10yyzzzz
            out[0] = 0xF0 | (u << 2) | (v >> 2);
            out[1] = 0x80 | ((v & 0x3) << 4) | w;
            out[2] = 0x80 | (x << 2) | (y >> 2);
            out[3] = 0x80 | ((y & 0x3) << 4) | z;
        }
    }
}
}
