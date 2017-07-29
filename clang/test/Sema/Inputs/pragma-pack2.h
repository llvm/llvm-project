
#ifndef NO_RECORD_2
struct S { int x; };
#endif

#ifdef SET_SECOND_HEADER
#pragma pack (8) // expected-note 2 {{previous '#pragma pack' directive that modifies alignment is here}}
#endif
