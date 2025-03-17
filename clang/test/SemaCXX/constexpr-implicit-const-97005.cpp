// RUN: %clang_cc1 -triple x86_64-linux-gnu %s

bool a;
constexpr const unsigned char c[] = { 5 };
constexpr const unsigned char d[1] = { 0 };
auto b = (a ? d : c);

constexpr const unsigned char c1[][1] = {{ 5 }};
constexpr const unsigned char d1[1][1] = {{ 0 }};
auto b1 = (a ? d1 : c1);
