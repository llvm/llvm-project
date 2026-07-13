// RUN: %clang_analyze_cc1 -std=c23 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_dump_ptr(const unsigned char *ptr);
void clang_analyzer_dump(unsigned char val);

int main() {
    const unsigned char SelfBytes[] = {
        #embed "embed.c"
    };
    clang_analyzer_dump_ptr(SelfBytes); // expected-warning {{&Element{SelfBytes,0 S64b,unsigned char}}}
    clang_analyzer_dump(SelfBytes[0]); // expected-warning {{47 U8b}}
}
