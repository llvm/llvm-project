#include "utf-8/neither-sjis-nor-ujis.h" // expected-error {{error opening file '<invalid buffer>'}}
// expected-error@* {{conversion from source encoding failed}}
constexpr auto North = –k;
constexpr auto South = “ì;
constexpr auto East = “Œ;
constexpr auto West = ¼;
