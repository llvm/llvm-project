#ifdef ALIGN_SET_HERE
#pragma align = mac68k
// expected-note@-1 {{previous '#pragma pack' directive that modifies alignment is here}}
// expected-warning@-2 {{unterminated '#pragma pack (push, ...)' at end of file}}
#endif

#ifdef RECORD_ALIGN
struct S {
  int x;
};
#endif
