
#include "hdr/types/wchar_t.h"

struct mbstate_t {
    wchar_t partial;
    unsigned char bits_processed;
    unsigned char total_bytes;
};
