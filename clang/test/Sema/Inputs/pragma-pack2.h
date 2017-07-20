
#ifdef SET_SECOND_HEADER
#pragma pack (8) // expected-note 2 {{previous '#pragma pack' directive that modifies alignment is here}}
#endif

struct S { int x; };
