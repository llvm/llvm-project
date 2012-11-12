// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix.Malloc -verify %s
// expected-no-diagnostics

typedef struct {
    char I[4];
    int S;
} Hdr;
typedef struct {
    short w;
} Hdr2;
typedef struct {
    Hdr2 usedtobeundef;
} Info;
typedef struct {
    const unsigned char *ib;
    int cur;
    int end;
} IB;
inline unsigned long gl(IB *input);
inline void gbs(IB *input, unsigned char *buf, int count);
void getB(IB *st, Hdr2 *usedtobeundef);
inline unsigned char gb(IB *input) {
    if (input->cur + 1 > input->end)
      ;
    return input->ib[(input->cur)++];
}
static void getID(IB *st, char str[4]) {
    str[0] = gb(st);
    str[1] = gb(st);
    str[2] = gb(st);
    str[3] = gb(st);
}
static void getH(IB *st, Hdr *header) {
    getID (st, header->I);
    header->S = gl(st);
}
static void readILBM(IB *st, Info *pic) {
    // Initialize field;
    pic->usedtobeundef.w = 5;

    // Time out in the function so that we will be forced to retry with no inlining.
    Hdr header;
    getH (st, &header);
    getID(st, header.I);
    int i = 0;
    while (st->cur < st->end && i < 4) {
      i++;
      getH (st, &header);
    }
}
int bitmapImageRepFromIFF(IB st, const unsigned char *ib, int il) {
    Info pic;
    st.ib = ib;
    st.cur = 0;
    st.end = il;
    readILBM(&st,&pic);
    return pic.usedtobeundef.w; // No undefined value warning here.
}
