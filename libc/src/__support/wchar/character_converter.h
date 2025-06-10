
#include "src/__support/wchar/mbstate.h"
#include "src/__support/wchar/utf_ret.h"
#include "hdr/types/wchar_t.h"

class CharacterConverter {
private:
    mbstate_t* state;

public:
    CharacterConverter();

    bool isComplete();

    int push(char utf8_byte);
    int push(wchar_t utf32);

    utf_ret<char> pop_utf8();
    utf_ret<wchar_t> pop_utf32();
};
